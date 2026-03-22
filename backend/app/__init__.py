from flask import Flask
from .config import DevConfig
from .extensions import db, migrate, cors,bcrypt,jwt
from .services.scheduler import start_scheduler
from .api.dashboard import dashboard_bp
from .api.analytics import analytics_bp
from .api.explorer import explorer_bp
from .api.auth import auth_bp 

def create_app(config_class=DevConfig):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Extensions
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})
    bcrypt.init_app(app)
    jwt.init_app(app)
    
    # Import models to ensure they are registered with SQLAlchemy
    from .models import video, clean_video, user, comment, accuracy, creator_profile

    # Blueprints
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    from .api.monitoring import monitoring_bp
    from .api.community import community_bp
    from .api.onboarding import onboarding_bp
    app.register_blueprint(monitoring_bp, url_prefix="/api/monitoring")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")
    app.register_blueprint(explorer_bp, url_prefix="/api/explorer")
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(community_bp, url_prefix="/api/community")
    app.register_blueprint(onboarding_bp, url_prefix="/api/onboarding")
    from .api.upload import upload_bp
    app.register_blueprint(upload_bp, url_prefix="/api/upload")

    # CLI Commands
    from .cli import register_commands
    register_commands(app)


    # Start the scheduler
    start_scheduler(app)

    return app
