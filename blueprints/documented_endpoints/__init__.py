from flask import Blueprint
from flask_restplus import Api

from blueprints.documented_endpoints.inference import inference

swagger = Blueprint('documented_api', __name__, url_prefix='/swagger')

api_extension = Api(
    swagger,
    title='Inference API',
    version='1.0',
    description='Inference application to calculate peoples\' face similarity',
)

api_extension.add_namespace(inference)
