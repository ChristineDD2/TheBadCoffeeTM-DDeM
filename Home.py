from ast import Call, ExceptHandler
from msilib import AMD64
from sys import exception, settrace
from cryptocom import api
import os
from ccxt import memory, trade, BadSymbol, NullResponse
import ccxt

class BitcoinCovidienOps:
    def ConnectionLostInSpace(self, trade, memory, exception):
        exchange = ccxt.cryptocom({
                                  "Connection-Power": 89,
                                  "Call-Host-Answer": True,
                                  "Block-Watch": True,
                                  "Safe-Blockchain": True,
                                  "AMD64-binary-gen-seeds": True
        })
                                  
        ConnectionLostInSpace(self=None, trade=0, memory="8GB", exception=False)
        return exchange




