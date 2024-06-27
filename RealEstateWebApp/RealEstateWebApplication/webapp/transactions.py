from django.db import models
from djongo import models
from django.utils import timezone
from djongo.models import Subquery, OuterRef
from webapp.owners import Owners
from django.db.models import Max , Count, Q
from django.http import JsonResponse
from django.http import HttpResponse , HttpResponseNotFound

class Transactions(models.Model):
    homeId = models.IntegerField()
    agentId = models.IntegerField()
    ownerId = models.CharField(max_length=100)
    seller = models.CharField(max_length=100)
    buyer=models.CharField(max_length=100)
    price=models.IntegerField()
    status=models.CharField(max_length=100)
    homeType=models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"Transaction: homeId={self.homeId}, agentId={self.agentId}, ownerId={self.ownerId}, seller={self.seller}, buyer={self.buyer}, price={self.price}, status={self.status}, homeType={self.homeType}"
    
    
    @staticmethod
    def save_transaction(homeid,agentid,ownerid,seller,buyer,price,status,htype):
        new_transaction=Transactions(
            homeId=homeid,
            agentId=agentid,
            ownerId=ownerid,
            seller=seller,
            buyer=buyer,
            price=price,
            status=status,
            homeType=htype,
            created_at=timezone.now()
        )
        
        saved_transaction=new_transaction.save()
        return saved_transaction
    
    @staticmethod
    def find_owners_who_dont_own_previous_homes():
        # Subquery to get the most recent buyer for each homeId
        try:
            
            
            distinct_home_ids = Transactions.objects.values('homeId').distinct()
            oldhomes={}
            for homeid in distinct_home_ids:
                all_prev_owners=[]
                print("homeId",homeid['homeId'])
                recent_trans=Transactions.objects.filter(homeId=homeid['homeId']).annotate(max_created_at=Max('created_at'))[:1]
                print("recnt_trans",recent_trans)
                new_owner=recent_trans[0].buyer
                print("new owner",new_owner)
                query= Q(seller__in=new_owner)
                all_trans=Transactions.objects.exclude(query).values('seller').distinct()
                for homeTrans in all_trans:
                    print("homeTrans",homeTrans.get('seller'))
                    old_owner=Owners.objects.get(SSN=homeTrans.get('seller'))
                    print("old_owner",old_owner)
            
                    all_prev_owners.append(old_owner)
                       
                    
                        # If the homeId doesn't exist, create a new list with the current owner
                oldhomes[homeid['homeId']]=all_prev_owners
            print("all_trans",oldhomes)
                
           
        except Exception as e:
            print(e)
            
        
        return oldhomes