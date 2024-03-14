from django.urls import path

from . import views
from .views import get_stock_names

urlpatterns = [
    path('', views.index, name='index'),
    path('get_stock_names/', get_stock_names, name='get_stock_names'),
]
