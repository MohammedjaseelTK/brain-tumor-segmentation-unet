from django.shortcuts import render

# Create your views here.
import os
import cv2

from rest_framework.views import APIView
from rest_framework.response import Response

from django.core.files.storage import default_storage

from model.predict import predict_mask


class PredictTumorView(APIView):

    def post(self, request):

        image = request.FILES.get("image")

        file_path = default_storage.save(
            image.name,
            image
        )

        full_path = default_storage.path(file_path)

        prediction = predict_mask(full_path)

        output_path = f"media/pred_{image.name}"

        cv2.imwrite(output_path, prediction)

        return Response({
            "prediction": output_path
        })