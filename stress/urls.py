from django.urls import path
from . import views


urlpatterns = [

    # Home page
    path(
        "",
        views.home,
        name="home"
    ),

    # Stress prediction page
    path(
        "prediction/",
        views.prediction,
        name="prediction"
    ),

    # Data mining dashboard
    path(
        "dashboard/",
        views.dashboard,
        name="dashboard"
    ),

]