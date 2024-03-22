from django.db import models

class Course(models.Model):
    courseName = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    
    class Meta:
        db_table = 'Course'
