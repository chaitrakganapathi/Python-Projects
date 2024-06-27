from django.db import models
from djongo import models
from webapp.homes import Homes


class Agents(models.Model):
    AgentID = models.IntegerField()
    Name = models.CharField(max_length=100)
    crate = models.IntegerField(db_column='CommissionRate')
    Company = models.CharField(max_length=100)
    
    def __str__(self):
        return f"Agent ID: {self.AgentID}, Name: {self.Name}, Commission Rate: {self.crate}, Company: {self.Company}"
    
    
    @staticmethod
    def save_new_agent(id,name,commission,company):
        # Create a new Homes object with the desired values
        new_agent = Agents(
            AgentID=id,
            Name=name,
            CommissionRate=commission,
            Company=company,
        )
        # Save the new_home object to the database
        saved_agent= new_agent.save()
    

    @staticmethod
    def display_agent():
        queryset=Agents.objects.filter()
        print(queryset.query.__str__())
        return queryset
    
    @staticmethod
    def get_commission(agentID):
        #home = Homes.objects.get(homeId=home_id)
        #print(home.agentID)
        # Fetch all homes sold by the agent
        #print(id)
        homes_sold_by_agent = Homes.objects.filter(agentID=agentID)
        agent = Agents.objects.get(AgentID=agentID)
        print(agent)
        # Initialize total commission earned by the agent
        total_commission = 0

        # Iterate over each home sold by the agent
        for home in homes_sold_by_agent:
        # Calculate commission based on the purchase price of the home and the commission rate
            commission = (home.price * agent.crate) / 100
        # Add commission to the total commission earned by the agent
            total_commission += commission
        print(total_commission)
        return total_commission
    