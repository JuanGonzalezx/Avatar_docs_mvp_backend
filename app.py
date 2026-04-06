from flask import Flask
from flask_cors import CORS

from controllers.avatar_controller import avatar_bp
from controllers.health_controller import health_bp
from services.database_service import init_db


def create_app():
    """Application factory."""
    app = Flask(__name__)

    # CORS for React frontend (dev + prod)
    CORS(app, origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ])

    # Register blueprints
    app.register_blueprint(avatar_bp)
    app.register_blueprint(health_bp)

    # Initialize database on startup
    with app.app_context():
        try:
            init_db()
        except Exception as e:
            print(f"[WARNING] Could not initialize DB: {e}")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
