from django.urls import path

from . import views
from .views import stock_dropdown
urlpatterns = [
    path('', views.index, name='index'),
    path('stock_dropdown/', views.stock_dropdown, name='stock_dropdown'),
]
