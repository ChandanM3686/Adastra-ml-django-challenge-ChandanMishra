
from django.urls import path
from .views import train_view, download_submission   

urlpatterns = [
    path('', train_view, name='train_view'),
    path('download/submission.csv', download_submission, name='download_submission'),
]
