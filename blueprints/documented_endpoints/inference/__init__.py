import werkzeug.datastructures
from flask_restplus import Namespace, Resource, fields

from werkzeug.datastructures import FileStorage

inference = Namespace('inference', 'Inference endpoints')

upload_model = inference.model("Upload", {
    "user_id": fields.Integer(
        required=True,
        description="Uploaded user id"
    )
})

compare_model = inference.model("Compare", {
    "score": fields.Float(
        required=True,
        description="score value"
    )
})

difference_result_model = inference.model("Difference result model", {
    "user_id": fields.Integer(
        required=True,
        description="Compared user"
    ),
    "distance": fields.Integer(
        required=True,
        description="Distance metric"
    )
})

difference_list_model = inference.model('Difference list model', {
    'entities': fields.Nested(
        difference_result_model,
        description='List of differences',
        as_list=True
    )
})

reference_upload_parser = inference.parser()
reference_upload_parser.add_argument('image', type=FileStorage, location='files')

preference_list_upload_parser = inference.parser()
preference_list_upload_parser.add_argument("file_1", type=FileStorage, location='files')
preference_list_upload_parser.add_argument("file_2", type=FileStorage, location='files')
preference_list_upload_parser.add_argument("file_etc", type=FileStorage, location='files')

compare_parser = inference.parser()
compare_parser.add_argument("reference_id", type=werkzeug.datastructures.string_types)
compare_parser.add_argument("compared_id", type=werkzeug.datastructures.string_types)


@inference.route("/upload/<int:user_id>", endpoint="with-parser")
class Upload(Resource):
    """Upload user's face to form a feature vector"""

    @inference.expect(reference_upload_parser)
    def post(self):
        """Upload user's face to form a feature vector"""

        return {
            "msg": "Upload successful"
        }, 201


@inference.route("/preference/<int:user_id>")
class SetPreference(Resource):
    """Set preference list"""

    @inference.expect(preference_list_upload_parser)
    def post(self):
        """Upload user's preference photos to find recommended profiles"""

        return {
            "msg": "Upload successful"
        }, 201
