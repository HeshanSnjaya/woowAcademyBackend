from django.urls import path
from . import views

urlpatterns = [
    path('predict-level', views.predict_level, name='predict_level'),
    path('get-cv-profiles/', views.get_cv_profiles, name='get_cv_profiles'),
    path('get-cv-profile-by-name/<str:username>/', views.get_cv_profile_by_username, name='get_cv_profile_by_username'),
    path('cv-profiles-by-level/<str:level>/', views.get_cv_profiles_by_level, name='get_cv_profiles_by_level'),
    path('get-top-10-names/', views.get_top_names, name='get_top_names'),
]