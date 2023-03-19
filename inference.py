import json
from threading import Thread
from typing import Union, Dict

import cv2
import face_recognition
import numpy as np
import pymongo
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager
from pymongo import MongoClient

app = Flask(__name__)
jwt = JWTManager(app)

client = MongoClient("mongodb://localhost:27017/")  # your connection string
db = client["murror"]
faces_collection = db["faces"]
face_distances_collection = db["face_similarities"]
BATCH_SIZE = 60
PAGE_SIZE = 10


def get_batch(collection, page_size, query=None, last_id=None):
    """Function returns `page_size` number of documents after last_id
    and the new last_id.
    """
    if last_id is None:
        # When it is first page
        cursor = collection.find().limit(page_size)
    else:
        cursor = collection.find({'_id': {'$gt': last_id}}).limit(page_size)

    if query:
        if last_id is None:
            # When it is first page
            cursor = collection.find(query).limit(page_size)
        else:
            cursor = collection.find({'_id': {'$gt': last_id}, **query}).limit(page_size)

    # Get the data
    data = [x for x in cursor]

    if not data:
        # No documents left
        return None, None

    # Since documents are naturally ordered with _id, last document will
    # have max id.
    last_id = data[-1]['_id']

    # Return data and last_id
    return data, last_id


def find_similarities(user_id: int):
    reference_encoding = faces_collection.find_one({"user_id": user_id})["encoding"]

    last_id = None
    batch, last_id = get_batch(
        collection=faces_collection,
        page_size=BATCH_SIZE,
        query={"user_id": {"$ne": user_id}},
        last_id=last_id)
    while batch is not None:
        face_ids = [el["user_id"] for el in batch]
        encodings = np.asarray([el["encoding"] for el in batch])
        similarities = face_recognition.face_distance(encodings, reference_encoding)
        face_similarities = [{"reference_id": user_id, "compared_id": compared_id, "distance": distance}
                             for compared_id, distance in zip(face_ids, similarities)]
        face_distances_collection.insert_many(face_similarities)
        batch, last_id = get_batch(
            collection=faces_collection,
            page_size=BATCH_SIZE,
            query={"user_id": {"$ne": user_id}},
            last_id=last_id)


@app.route('/upload/<int:user_id>', methods=['POST'])
def upload(user_id: int):
    raw_image = request.files['face'].read()
    npimg = np.fromstring(raw_image, np.uint8)
    face_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    face_encodings = face_recognition.face_encodings(face_image)

    if len(face_encodings) > 1:
        return jsonify({"msg": "More than one face identified"}), 400

    faces_collection.insert_one({"user_id": user_id, "encoding": face_encodings[0].tolist()})
    Thread(target=find_similarities, args=(user_id,)).run()

    return jsonify({"msg": "Upload sucessful"}), 201


@app.route('/compare', methods=['GET'])
def compare():
    try:
        content = request.get_json()
        reference_id = content["reference_id"]
        compared_id = content["compared_id"]

        assert isinstance(reference_id, int)
        assert isinstance(compared_id, int)

        distance = face_distances_collection.find_one({ "$or": [
            {"reference_id": reference_id, "compared_id": compared_id},
            {"reference_id": compared_id, "compared_id": reference_id}
        ]})["distance"]

        result = {"score": distance}
        return jsonify({"msg": "Success", "result": result}), 200

    except AssertionError:
        return jsonify({"msg": "Incorrect request format"}), 400


@app.route('/distance/<int:user_id>/<int:page_num>', methods=["GET"])
def get_similar_users(user_id: int, page_num: int):
    try:
        results_amount = face_distances_collection.count_documents({"$or": [
            {"reference_id": user_id},
            {"compared_id": user_id}
        ]})
        assert page_num * PAGE_SIZE < results_amount

        result = face_distances_collection.find({"$or": [
            {"reference_id": user_id},
            {"compared_id": user_id}
        ]}).sort("distance", pymongo.ASCENDING).skip(PAGE_SIZE * page_num).limit(PAGE_SIZE)

        def get_compared_id(el: Dict):
            return el["reference_id"] if el["reference_id"] != user_id else el["compared_id"]

        result = [{"user_id": get_compared_id(el), "distance": el["distance"]} for el in result]

        return jsonify({"msg": "Success", "result": result}), 200

    except Union[AssertionError]:
        return jsonify({"msg": "Incorrect page number"}), 400


if __name__ == '__main__':
    app.run(debug=True)
