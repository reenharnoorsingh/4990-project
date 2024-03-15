from django.urls import path

from . import views

from .views import stock_chart, submit_stock_form

urlpatterns = [
    path('', views.index, name='index'),
    path('get_stock_names/', views.get_stock_names, name='get_stock_names'),
    path('stock_chart/', stock_chart, name='stock_chart'),
    path('submit_stock_form/', submit_stock_form, name='submit_stock_form'),
]

