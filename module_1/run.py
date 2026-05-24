# Entry point — run this file with: python run.py
from app import create_app

# Build the Flask application using the factory function in app/__init__.py
app = create_app()

if __name__ == '__main__':
    # Start the development server on all interfaces at port 8080
    # debug=True auto-reloads when source files change
    app.run(host='0.0.0.0', port=8080, debug=True)
