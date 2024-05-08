from ..abstract import AbstractAnonymization


class AbstractFaceAnonymization(AbstractAnonymization):
    name = "abstractface"

    def get_faces(self, img):
        if not img.bbox:
            import face_recognition

            image = face_recognition.load_image_file(img.get_path())
            return face_recognition.face_locations(image)
        else:
            return [(img.bbox["top"], img.bbox["right"], img.bbox["bottom"], img.bbox["left"])]
