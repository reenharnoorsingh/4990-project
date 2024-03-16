from django.urls import path

from . import views

from .views import submit_stock_form, stock_chart

urlpatterns = [
    path('', views.index, name='index'),
    path('get_stock_names/', views.get_stock_names, name='get_stock_names'),
    path('submit_stock_form/', submit_stock_form, name='submit_stock_form'),
    path('stock_chart/', stock_chart, name='stock_chart'),

]

