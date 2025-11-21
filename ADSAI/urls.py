from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('ADSTRA.urls')),  # connects your ADSTRA app
    path('admin/', admin.site.urls),
]
