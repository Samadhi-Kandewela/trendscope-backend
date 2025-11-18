from flask import Flask
from .config import DevConfig
from .extensions import db, migrate, cors,bcrypt,jwt
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

    # Blueprints
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")
    app.register_blueprint(explorer_bp, url_prefix="/api/explorer")
    app.register_blueprint(auth_bp, url_prefix="/api/auth")

    return app
