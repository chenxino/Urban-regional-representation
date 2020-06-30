from django.urls import path
 
from . import views

app_name = 'blog'
urlpatterns = [
    path('', views.index, name='index'),
    path('posts/<int:pk>/', views.detail, name='detail'),
    path('map', views.map, name='map'),
    path('map/<int:id>', views.map_id, name='map_id'),
    path('poi', views.poi, name='poi'),
    path('taxi', views.taxi, name='taxi'),
]