from django.conf.urls import url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^$', views.start),
    url(r'^page2/$', views.page2),
    url(r'^page3/$', views.page3),
    url(r'^takeinput/$', views.takeinput),
]
