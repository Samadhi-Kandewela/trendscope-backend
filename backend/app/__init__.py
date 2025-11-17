from flask import Flask
from .config import DevConfig
from .extensions import db, migrate, cors
from .api.dashboard import dashboard_bp
from .api.analytics import analytics_bp
from .api.explorer import explorer_bp

def create_app(config_class=DevConfig):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Extensions
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})

    # Blueprints
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")
    app.register_blueprint(explorer_bp, url_prefix="/api/explorer")
    
    return app
