from agents.fql import FQLAgent
from agents.gsflow import GSflow
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    fql=FQLAgent,
    gsflow=GSflow,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
