from django.urls import path

from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('stock_dropdown1', views.stock_dropdown1, name='stock_dropdown1'),
    path('stock_dropdown2', views.stock_dropdown2, name='stock_dropdown2'),
    path('stock_display', views.stock_display, name='graph_display'),


]
