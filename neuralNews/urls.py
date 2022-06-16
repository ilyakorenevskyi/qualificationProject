from django.conf.urls.static import static
from django.urls import path
from neuralNews import views
from qualificationProject import settings

urlpatterns = [
    path('', views.main_page, name='main_page'),
    path('news_generation', views.news_generation, name='news_generation'),
    path('news_classification', views.news_classification, name='news_classification'),
    path('feedback', views.feedback, name='feedback'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
