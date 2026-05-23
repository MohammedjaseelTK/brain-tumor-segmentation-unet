from django.urls import path

from .views import PredictTumorView


urlpatterns = [

    path(
        'predict/',
        PredictTumorView.as_view(),
        name='predict'
    ),
]