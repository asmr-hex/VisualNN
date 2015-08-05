# VisualNN
# A didactic tool for visualizing simple artificial neural networks and 
# how they learn.
#
# MIT copyright tuntun, July 2015 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import re

class NeuralNetwork(object):
	"""A general neural network supporting a variety of architectures. 
	
	The network class provides the architecture of configured neurons 
	as well as the processes with which to operate on them.
	"""
	
	__Layers = {}	#Neuronal layers
	__Weights = []	#Network Weights
	__illustrator = None	#Illustrator instance
	def __init__(self, architecture):
		"""Inits NeuralNetwork with architecture."""
		self.__buildArchitecture(architecture)
		self.__illustrator = Illustrator(self.__Layers)
		
	def __buildArchitecture(self, architecture):
		"""Parses architecture string and builds network."""
		L_GROUP = 1	#Layer group for regex
		S_GROUP = 4	#NodeID start group for regex
		E_GROUP = 5	#NodeID end group for regex
		chains = re.split(r',(?=\(\d+,\d*:?\d+\))', architecture)
		rexp = re.compile(r'\((\d+),(((\d*):)?(\d*))\)')
		for chain in chains:	#Parse each defined chain
			chain = re.split(r'->',chain)
			for i in xrange(0,len(chain)-1):
				src_node = rexp.search(chain[i])	#Get source neuron(s)
				src_layer = int(src_node.group(L_GROUP))
				src_ids = self.__parseRange(src_node.group(S_GROUP),src_node.group(E_GROUP))
				src_neurons = self.__getNeurons(src_layer, src_ids)

				dest_node = rexp.search(chain[i+1])	#Get destination neuron(s)
				dest_layer = int(dest_node.group(L_GROUP))
				dest_ids = self.__parseRange(dest_node.group(S_GROUP),dest_node.group(E_GROUP))
				dest_neurons = self.__getNeurons(dest_layer, dest_ids)
				for src_neuron in src_neurons:
					src_neuron.addOutput(dest_neurons)	#Add dests to src
				for dest_neuron in dest_neurons:
					dest_neuron.addInput(src_neurons)	#Add src to dests
	
	def __parseRange(self, start, end):
		"""Helper function to parse ranges for architecture setup"""
		end = int(end)
		if start is None:
			start = end
		else:
			start = int(start)
		return range(start, end+1)

	def __getNeurons(self, layer, nids):
		"""Helper function to find neuron, or return a new one"""
		neurons = []
		for nid in nids:
			layers = self.__Layers.keys()
			if layer not in layers:
				neuron = Neuron(layer, nid)
				self.__Layers[layer] = [neuron]
				neurons.append(neuron)
			else:
				neuron_ids = list(map((lambda x: x.getID()),self.__Layers[layer]))
				if nid in neuron_ids:
					neurons.append(self.__Layers[layer][neuron_ids.index(nid)])
				else:
					neuron = Neuron(layer, nid)
					self.__Layers[layer].append(neuron)
					neurons.append(neuron)
		return neurons
				
		
		
	def feedForward(self, x):
		"""Feeds network inputs, stores intermediate values, returns output."""
		pass
		
	def backPropagation(self):
		"""Performs backpropogation of errors over the neural network."""
		pass
		
		
class Neuron(object):
	"""A single computational neuron.
	
	The neuron class provides an activation function for inputs, 
	storage for activations achieved during forward passes, and
	storage for error signals during backwards propogation. """
	
	__f = ""	#Activation function
	__A = 0	#Current activation
	__inputs = []	#list of input neurons
	__outputs = []	#list of output neurons
	__layer = None
	__id = None
	def __init__(self, layer, nid):
		"""Inits Neuron with activation function."""
		self.__layer = layer
		self.__id = nid
		
	def computeActivation(self, x):
		"""Uses activation function to compute activation."""
		pass
	
	def getActivation(self):
		"""Gets current activation from neuron."""
		return self.__A
		
	def addOutput(self, neurons):
		"""Adds output connectivity between this and other neurons."""
		if type(neurons) is list:
			self.__outputs =  self.__outputs + neurons
		else:
			self.__outputs.append(neurons)
		
	def addInput(self, neurons):
		"""Adds input connectivity between this and other neurons."""
		if type(neurons) is list:
			self.__inputs = self.__inputs + neurons
		else:
			self.__inputs.append(neurons)
			
	def getOutput(self):
		"""Gets output connected neurons."""
		return self.__outputs
		
	def getID(self):
		"""Get the neuron id of this neuron."""
		return self.__id
	
	def getLayer(self):
		"""Get the layer in which this neuron resides."""
		return self.__layer
		
		
class Illustrator(object):
	"""Illustration tool for visualizing neural networks.
	
	Instantiated within a NeuralNetwork object and called from within."""
	
	__on = False	#Illustrator on/off
	__fig = plt.figure()	#Illustrator figure handles
	__ax = __fig.gca()		#Illustrator axes
	__nodes = []			#Graphical nodes
	__layer_colors = []		#Layer color
	__layer_x = []
	
	def __init__(self, layers):
		"""Inits visualization tool for a NeuralNetwork instance."""
		#self.__ax.plot(np.arange(1,10), np.arange(2,11))
		
		#get max nodes in layers
		maxNodes = max([len(layers[k]) for k in layers.keys()])
		L = len(layers)
		self.__layer_x = np.linspace(0, 10, L)
		for l in xrange(0,L):
			y = 10./(len(layers[l])+1)
			for v in xrange(0, len(layers[l])):
				circle = mpatches.Circle([self.__layer_x[l], y*(v+1)], 0.4,ec="none")
				self.__nodes.append(circle)
				#plot lines
				outs = layers[l][v].getOutput()
				for o in xrange(0, len(outs)):
					lprime = outs[o].getLayer()
					yprime = 10./(len(layers[lprime])+1)
					xplt = [self.__layer_x[l], self.__layer_x[lprime]]
					yplt = [y*(v+1), yprime*(o+1)]
					self.__ax.plot(xplt,yplt, zorder=1)
					
		self.__layer_colors = np.linspace(0, 1, L)
		collection = PatchCollection(self.__nodes, cmap=plt.cm.hsv, alpha=0.9, zorder = 2)
		collection.set_array(np.array(self.__layer_colors))
		
		self.__ax.add_collection(collection)
		self.__ax.set_ylim(0,10)
		self.__ax.set_xlim(0,10)
		self.__ax.axis('equal')
		plt.show()
		
	def draw(self):
		"""Draws the current NeuralNetwork instance state."""
		pass
		
		
		
		
		
	
	
	