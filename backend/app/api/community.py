from flask import Blueprint, jsonify, request
from datetime import datetime
from ..extensions import db
from ..models.community import Discussion, DiscussionComment

community_bp = Blueprint('community', __name__)

# ---------------------------------------------------------------------------
# DISCUSSIONS (The Feed)
# ---------------------------------------------------------------------------

@community_bp.route('/discussions', methods=['GET'])
def get_discussions():
    """
    Get list of discussions.
    Query Params:
      - tag: Filter by tag (e.g. "gaming")
      - sort: "recent" (default) or "popular" (most upvotes)
    """
    tag = request.args.get('tag')
    sort = request.args.get('sort', 'recent')
    
    query = Discussion.query
    
    # 1. Filter by ID (descending) roughly equals "recent".
    #    Real "recent" uses created_at.
    if sort == 'popular':
        query = query.order_by(Discussion.upvotes.desc())
    else:
        query = query.order_by(Discussion.created_at.desc())
        
    discussions = query.all()
    
    # Python-side filtering for JSON tags (SQLite/Postgres JSON compat vary, doing safely)
    if tag:
        tag = tag.lower()
        discussions = [d for d in discussions if d.tags and tag in [t.lower() for t in d.tags]]
        
    return jsonify([d.to_dict() for d in discussions])

@community_bp.route('/discussions', methods=['POST'])
def create_discussion():
    """
    Create a new discussion thread (or "Post").
    Body: {
        "title": "...",
        "body": "...",
        "tags": ["..."],
        "media_url": "...",     # Optional: Graph/Image
        "linked_context": {...} # Optional: Deep-link context
    }
    """
    data = request.get_json()
    if not data or not data.get('title'):
        return jsonify({"error": "Title is required"}), 400
        
    new_post = Discussion(
        title=data['title'],
        body=data.get('body', ''),
        tags=data.get('tags', []),
        media_url=data.get('media_url'),
        linked_context=data.get('linked_context'),
        author_username=data.get('author_username', 'Anonymous')
    )
    
    db.session.add(new_post)
    db.session.commit()
    
    return jsonify(new_post.to_dict()), 201

@community_bp.route('/discussions/<int:id>', methods=['GET'])
def get_discussion_detail(id):
    """
    Get full details of a discussion + threaded comments.
    """
    post = Discussion.query.get_or_404(id)
    
    # Increment view count
    post.view_count += 1
    db.session.commit()
    
    # Get comments
    comments = [c.to_dict() for c in post.comments]
    
    # Return matched structure
    resp = post.to_dict()
    resp['comments'] = comments
    return jsonify(resp)

# ---------------------------------------------------------------------------
# COMMENTS & INTERACTIONS
# ---------------------------------------------------------------------------

@community_bp.route('/discussions/<int:id>/comments', methods=['POST'])
def add_comment(id):
    """
    Add a comment to a discussion.
    """
    post = Discussion.query.get_or_404(id)
    data = request.get_json()
    
    if not data or not data.get('body'):
        return jsonify({"error": "Comment body is required"}), 400
        
    new_comment = DiscussionComment(
        discussion_id=post.id,
        body=data['body'],
        author_username=data.get('author_username', 'Anonymous')
    )
    
    db.session.add(new_comment)
    db.session.commit()
    
    return jsonify(new_comment.to_dict()), 201

@community_bp.route('/discussions/<int:id>/vote', methods=['POST'])
def vote_discussion(id):
    """
    Upvote (or Downvote) a discussion.
    Body: {"type": "up" | "down", "username": "UserA"}
    """
    post = Discussion.query.get_or_404(id)
    data = request.get_json()
    
    vote_type = data.get('type', 'up')
    username = data.get('username')
    
    if not username:
        return jsonify({"error": "Username is required to vote"}), 400
        
    from ..models.community import DiscussionVote
    
    # Check existing vote
    existing_vote = DiscussionVote.query.filter_by(
        discussion_id=post.id, 
        user_username=username
    ).first()
    
    if existing_vote:
        # Case 1: Same vote type -> Toggle OFF (Remove vote)
        if existing_vote.vote_type == vote_type:
            db.session.delete(existing_vote)
            if vote_type == 'up':
                post.upvotes -= 1
            else:
                post.upvotes += 1 # Determine logic for downvotes later, assuming just upvotes
            action = "removed"
            
        # Case 2: Different vote type -> Switch (Not implemented fully for simple upvotes, treating as toggle)
        else:
            # For now, let's just update type and score
            if vote_type == 'up':
                post.upvotes += 2 # -1 (old down) + 1 (new up) = +2 diff? No:
                # If was down (-1), now up (+1), change is +2.
                # But our simpler model might not support downvotes fully yet.
                # Let's keep it simple: Just toggle for now.
                existing_vote.vote_type = vote_type
                # post.upvotes logic dependent on implementation.
                pass 
            # Reverting: Simple implementation first.
            # If exist -> return 400 or just toggle off.
            # Let's just TOGGLE OFF if exists regardless of type for MVP simplicity
            db.session.delete(existing_vote)
            post.upvotes -= 1
            action = "removed"

    else:
        # Case 3: New Vote
        new_vote = DiscussionVote(
            discussion_id=post.id,
            user_username=username,
            vote_type=vote_type
        )
        db.session.add(new_vote)
        if vote_type == 'up':
            post.upvotes += 1
        elif vote_type == 'down':
            post.upvotes -= 1
            
        action = "added"
        
    db.session.commit()
    
    return jsonify({
        "id": post.id, 
        "upvotes": post.upvotes,
        "action": action
    })

@community_bp.route('/strategists', methods=['GET'])
def get_top_strategists():
    """
    Get top authors based on total upvotes received.
    Returns: [{"username": "Sarah", "karma": 150}, ...]
    """
    from sqlalchemy import func
    
    # Aggregation: Group by author -> Sum(upvotes)
    results = db.session.query(
        Discussion.author_username,
        func.sum(Discussion.upvotes).label('total_upvotes')
    ).group_by(Discussion.author_username).order_by(func.sum(Discussion.upvotes).desc()).limit(5).all()
    
    # Format
    strategists = [
        {"username": r[0], "karma": int(r[1]) if r[1] else 0}
        for r in results
    ]
    
    return jsonify(strategists)

@community_bp.route('/tags', methods=['GET'])
def get_trending_tags():
    """
    Get top tags based on usage count across all discussions.
    Returns: [{"tag": "gaming", "count": 25}, ...]
    """
    from collections import Counter
    
    # Fetch all tags (JSON list)
    # Note: For SQLite/Postgres compatibility without complex JSON functions,
    # we fetch all and aggregate in Python. For <10k posts this is fine.
    discussions = db.session.query(Discussion.tags).all()
    
    # Flatten list: [[t1, t2], [t2, t3]] -> [t1, t2, t2, t3]
    all_tags = []
    for d in discussions:
        if d.tags: # d.tags is a Python list
            all_tags.extend([t.lower() for t in d.tags])
            
    # Count
    counts = Counter(all_tags)
    
    # Get Top 10
    top_tags = [
        {"tag": tag, "count": count}
        for tag, count in counts.most_common(10)
    ]
    
    return jsonify(top_tags)
