from ...lib.result import Result
from .abstract import AbstractFacePrivacy
from ...lib.inference import Classification

import botocore
import boto3


class RekognitionClassification(Classification, AbstractFacePrivacy):
    """Use AWS Rekognition as the privacy method.
    Rekognition documentation: https://docs.aws.amazon.com/rekognition/?id=docs_gateway
    AWS Python SDK documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
    AWS Python SDK Rekognition documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rekognition.html

    Required pips:
        - boto3

    This will require that you set up your AWS account credentials (and default region) as described here:
        https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration

    Parameters:
        - opt

    Options: (Parameters["opt"]; do not influece output)
        - (bool) cleanup: whether to delete training collection from AWS after privacy


    """

    def validate_config(self):
        if "opt" not in self.config or "cleanup" not in self.config["opt"]:
            self.config["opt"]["cleanup"] = True

    def enroll(self, set):
        self.training_set = set
        self.log.info("Starting privacy.\n\tFolder: " + set.folder + "\n\tConfiguration: " + str(self.config))

        if "arn" in set.meta and set.meta["arn"] not in [None, "", False]:
            if set.meta["arn"].split("/")[-1] == set.name:
                self.log.info("Upload skipped. Dataset already on AWS.")
                self.collection = set.name
            else:
                raise AttributeError("ARN does not match.")
        else:
            client = boto3.client("rekognition")

            try:
                res = client.create_collection(CollectionId=set.name)
            except botocore.exceptions.ClientError as err:
                if err.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                    self.log.warn(
                        "Collection already exists, but not known to Dataset metadata. Assuming error during creation or indexing. Recreating."
                    )
                    client.delete_collection(CollectionId=set.name)
                    res = client.create_collection(CollectionId=set.name)
                else:
                    raise err
            arn = res["CollectionArn"]

            for image_id, image in set.datapoints.items():
                with open(image.get_path(), "rb") as image_file:
                    image_data = image_file.read()

                client.index_faces(
                    CollectionId=set.name,
                    DetectionAttributes=["DEFAULT"],
                    ExternalImageId=image_id,
                    MaxFaces=1,
                    Image={
                        "Bytes": image_data,
                    },
                )
            self.collection = set.name
            set.meta["arn"] = arn
            set.save_meta()

    def classify_point(self, image):
        client = boto3.client("rekognition")
        rs = Result(image.idname, image.pointname)
        res = "Result for " + image.pointname + ":"
        res += " correct: " + image.idname + " recognized:"

        with open(image.get_path(), "rb") as image_file:
            image_data = image_file.read()

        response = client.search_faces_by_image(
            CollectionId=self.collection,
            Image={
                "Bytes": image_data,
            },
            FaceMatchThreshold=0.0,
        )

        for match in response["FaceMatches"]:
            dist = 1 - (match["Similarity"] / 100)
            img = match["Face"]["ExternalImageId"]

            res += " " + img + " (" + str(round(dist, 4)) + ");"
            rs.add_recognized(img.split(".")[0], dist=dist)

        self.log.debug(res)
        return rs

    def cleanup(self):
        if not self.config["opt"]["cleanup"]:
            client = boto3.client("rekognition")
            client.delete_collection(CollectionId=self.collection)
            del self.training_set.meta["arn"]
            self.training_set.save_meta()
