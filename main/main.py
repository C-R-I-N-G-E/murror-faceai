from waitress import serve
from inference import app as my_app  # renamed to distinguish from waitress' 'app'

if __name__ == "__main__":
    serve(my_app, host="0.0.0.0", port=5000)
