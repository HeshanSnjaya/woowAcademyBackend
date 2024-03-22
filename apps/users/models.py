from django.core.validators import MinValueValidator
from django.db import models


class Student(models.Model):
    studentID = models.CharField(max_length=50, primary_key=True)
    firstName = models.CharField(max_length=255)
    lastName = models.CharField(max_length=255)
    userName = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    phone = models.CharField(max_length=255)
    dob = models.DateField(default='2006-01-01')
    userPassword = models.CharField(max_length=255)
    level = models.CharField(max_length=100, default='pending')

    class Meta:
        db_table = 'Student'


class Address(models.Model):
    addressID = models.CharField(max_length=50, primary_key=True)
    userID = models.ForeignKey(Student, on_delete=models.CASCADE, db_column='userID')
    lineOne = models.CharField(max_length=255)
    lineTwo = models.CharField(max_length=255)
    city = models.CharField(max_length=255)
    postCode = models.CharField(max_length=255)

    class Meta:
        db_table = 'Address'


class Recruiter(models.Model):
    recruiterID = models.CharField(max_length=50, primary_key=True, db_column='recruiterID')
    firstName = models.CharField(max_length=255)
    lastName = models.CharField(max_length=255)
    userName = models.CharField(max_length=255)
    profile_img = models.TextField()
    email = models.CharField(max_length=255)
    phone = models.CharField(max_length=255)
    recruiterPassword = models.CharField(max_length=255)

    class Meta:
        db_table = 'Recruiter'


class CV_Profile(models.Model):
    cvID = models.CharField(max_length=50, primary_key=True)
    studentID = models.ForeignKey(Student, on_delete=models.CASCADE, db_column='studentID')
    profile_img = models.TextField()
    about = models.TextField()
    points = models.IntegerField()

    class Meta:
        db_table = 'CV_Profile'


class Objective(models.Model):
    objectiveID = models.CharField(max_length=50, primary_key=True)
    cvID = models.ForeignKey(CV_Profile, on_delete=models.CASCADE, db_column='cvID')
    objective_description = models.TextField()

    class Meta:
        db_table = 'Objective'


class Education(models.Model):
    educationID = models.CharField(max_length=50, primary_key=True)
    cvID = models.ForeignKey(CV_Profile, on_delete=models.CASCADE, db_column='cvID')
    institution = models.CharField(max_length=255)
    education_course = models.CharField(max_length=255)
    education_start_date = models.DateField()
    education_end_date = models.DateField()

    class Meta:
        db_table = 'Education'


class Skill(models.Model):
    skillID = models.CharField(max_length=50, primary_key=True)
    cvID = models.ForeignKey(CV_Profile, on_delete=models.CASCADE, db_column='cvID')
    skill_name = models.CharField(max_length=255)
    skill_level = models.CharField(max_length=255)

    class Meta:
        db_table = 'Skill'


class SocialMedia(models.Model):
    socialMediaID = models.CharField(max_length=50, primary_key=True)
    cvID = models.ForeignKey(CV_Profile, on_delete=models.CASCADE, db_column='cvID')
    socialMedia_name = models.CharField(max_length=255)
    socialMedia_link = models.CharField(max_length=255)

    class Meta:
        db_table = 'SocialMedia'


class WorkExperience(models.Model):
    workExperienceID = models.CharField(max_length=50, primary_key=True)
    cvID = models.ForeignKey(CV_Profile, on_delete=models.CASCADE, db_column='cvID')
    company_name = models.CharField(max_length=255)
    job_title = models.CharField(max_length=255)
    job_start_date = models.DateField()
    job_end_date = models.DateField()
    job_description = models.TextField()
    job_address = models.CharField(max_length=255)
    duration = models.DurationField(null=True, blank=True)

    class Meta:
        db_table = 'WorkExperience'

    def save(self, *args, **kwargs):
        if self.job_start_date and self.job_end_date:
            self.duration = self.job_end_date - self.job_start_date
        super().save(*args, **kwargs)


class VolunteerExperience(models.Model):
    volunteerExperienceID = models.CharField(max_length=50, primary_key=True)
    cvID = models.ForeignKey(CV_Profile, on_delete=models.CASCADE, db_column='cvID')
    organization_name = models.CharField(max_length=255)
    role = models.CharField(max_length=255)
    volunteer_start_date = models.DateField()
    volunteer_end_date = models.DateField()
    volunteer_description = models.TextField()

    class Meta:
        db_table = 'VolunteerExperience'


class Project(models.Model):
    projectID = models.CharField(max_length=50, primary_key=True)
    cvID = models.ForeignKey(CV_Profile, on_delete=models.CASCADE, db_column='cvID')
    project_name = models.CharField(max_length=255)
    project_description = models.TextField()

    class Meta:
        db_table = 'Project'
