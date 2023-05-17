from threading import Thread

import cv2
import face_recognition
import numpy as np
from flask import request, jsonify, Blueprint
import psycopg2
from loguru import logger
from pymongo import MongoClient

try:
    conn = psycopg2.connect('postgresql://postgres:qwerty@localhost:5432/murror')
    logger.info("Connected to database")
except Exception as e:
    logger.error("Can't connect to database")
    logger.error(e)

client = MongoClient("mongodb://localhost:27017/")
db = client["murror"]
faces_collection = db["faces"]
preferences_collection = db["preferences"]
BATCH_SIZE = 60
RECOMMENDED_PAGE_SIZE = 10

inference = Blueprint('inference', __name__)


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
    preference_encoding_list = preferences_collection.find_one({"user_id": user_id})["encoding"]

    last_id = None
    batch, last_id = get_batch(
        collection=faces_collection,
        page_size=BATCH_SIZE,
        query={"user_id": {"$ne": user_id}},
        last_id=last_id)
    cursor = conn.cursor()
    while batch is not None:
        face_ids = [el["user_id"] for el in batch]
        encodings = np.asarray([el["encoding"] for el in batch])
        similarities = [max(face_recognition.face_distance(preference_encoding_list, potential_match))
                        for potential_match in encodings]
        face_similarities = [(user_id, compared_id, distance)
                             for compared_id, distance in zip(face_ids, similarities)]
        args = ','.join(cursor.mogrify("(%s,%s,%s)", i).decode('utf-8')
                        for i in face_similarities)
        print(face_similarities)
        cursor.execute("INSERT INTO face_similarities (reference_id, compared_id, distance) VALUES " + args)
        conn.commit()
        logger.info(f"Added {cursor.rowcount} new rows")
        batch, last_id = get_batch(
            collection=faces_collection,
            page_size=BATCH_SIZE,
            query={"user_id": {"$ne": user_id}},
            last_id=last_id)
    cursor.close()

@inference.route('/upload/<int:user_id>', methods=['POST'])
def upload(user_id: int):
    raw_image = request.files['face'].read()
    npimg = np.fromstring(raw_image, np.uint8)
    face_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    face_encodings = face_recognition.face_encodings(face_image)

    if len(face_encodings) > 1:
        return jsonify({"msg": "More than one face identified"}), 400

    faces_collection.insert_one({"user_id": user_id, "encoding": face_encodings[0].tolist()})

    return jsonify({"msg": "Upload successful"}), 201


@inference.route("/preference/<int:user_id>", methods=["POST"])
def set_preference(user_id: int):
    raw_image_list = list(request.files.values())
    npimg_list = [np.fromstring(raw_image.read(), np.uint8) for raw_image in raw_image_list]
    face_image_list = [cv2.imdecode(npimg, cv2.IMREAD_COLOR) for npimg in npimg_list]

    face_encoding_list = [face_recognition.face_encodings(face_image) for face_image in face_image_list]

    if any(map(lambda f: len(f) > 1, face_encoding_list)):
        return jsonify({"msg": "More than one face identified"}), 400

    face_encoding_list = [face_encoding[0].tolist() for face_encoding in face_encoding_list]

    preferences_collection.insert_one({"user_id": user_id, "encoding": face_encoding_list})

    # FIXME: make process run separate from API
    Thread(target=find_similarities, args=(user_id,)).run()

    return jsonify({"msg": "Upload successful"}), 201