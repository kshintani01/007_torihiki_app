from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('single/', views.single_input, name='single_input'),
    path('csv/', views.csv_upload, name='csv_upload'),
    path('debug-form/', views.debug_form, name='debug_form'),
]