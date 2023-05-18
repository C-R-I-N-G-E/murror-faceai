from flask import Flask

from blueprints.documented_endpoints import swagger
from blueprints.inference import inference

app = Flask(__name__)
app.register_blueprint(inference)
app.register_blueprint(swagger)
