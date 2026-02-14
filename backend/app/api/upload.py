from flask import Blueprint, request, jsonify, url_for, current_app
import os
import secrets
from werkzeug.utils import secure_filename

upload_bp = Blueprint('upload', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route('/', methods=['POST'])
def upload_file():
    """
    Upload a file (Image only).
    Key: 'file'
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        # 1. Secure filename & add random component to prevent collision
        filename = secure_filename(file.filename)
        random_hex = secrets.token_hex(8)
        _, f_ext = os.path.splitext(filename)
        new_filename = f"{random_hex}{f_ext}"
        
        # 2. Ensure directory exists
        upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        # 3. Save
        file_path = os.path.join(upload_folder, new_filename)
        file.save(file_path)
        
        # 4. Return correct URL
        # Assuming app is running at root, url_for('static', ...) generates /static/...
        # In production this might differ, but for local dev this works.
        file_url = url_for('static', filename=f'uploads/{new_filename}', _external=True)
        
        return jsonify({"url": file_url}), 201
        
    return jsonify({"error": "File type not allowed"}), 400
