from django.urls import path
from . import views

# URLConf
urlpatterns = [
    # create new user
    path('create', views.createUser),
    #check if user exists
    path('check', views.checkValidUser),
    # view user profile by id
    path('profile/<id>', views.getUserProfile),
    # view all users
    path('all', views.viewAllUsers),
    # update user profile
    path('update', views.updateUser),
    # delete user profile
    path('delete/<id>', views.deleteUser),
    # update student points
    path('points/<id>/<points>', views.updateStudentPoints),
    
    # create cv profile
    path('cv/create', views.createStudentCVProfile),
    
     # view student cv profile by student id
    path('cv/<id>', views.getStudentCVProfile),
    path('get_cv_ids/', views.get_cv_ids, name='get_cv_ids'),
]