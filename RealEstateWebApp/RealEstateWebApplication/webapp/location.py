from django.db import models
from djongo import models



class Address(models.Model):
    street = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=2)
    zipcode = models.CharField(max_length=5)
    

    class Meta:
        abstract = True  # Makes this model abstract, so Djongo won't create a separate table

class Location(models.Model):
    home_ID = models.IntegerField(unique=True)
    unit_number = models.CharField(max_length=10, null=True, blank=True)
    address = models.EmbeddedField(
        model_container=Address,
    )

    def __str__(self):
        address = self.address
        return f"Location {self.home_ID}: {address.street}, {address.city}, {address.state}, {address.zipcode}"
