# -*- coding: latin-1 -*-

import numpy as np
import contextlib
import random

NOGRAD = False

@contextlib.contextmanager
def no_grad():
	global NOGRAD
	old = NOGRAD
	NOGRAD = True
	try:
		yield
	finally:
		NOGRAD = old

def detach(u: "Unity"):
	"""Retourne une copie sans lien au graphe"""
	return Unity(u.unity)   # nouvelle Unity, sans _prev


class Unity:

	def __init__(self, unity, _children=()):
		self.unity = unity
		self.grad = 0
		# internal variables used for autograd graph construction
		#self._backward = lambda: None
		#self._prev = set(_children)
		if NOGRAD:
			self._prev = set()         # pas de dépendances
			self._backward = lambda: None
		else:
			self._prev = set(_children)
			self._backward = lambda: None

	def __add__(self, other):
		other = other if isinstance(other, Unity) else Unity(other)
		out = Unity(self.unity + other.unity, (self, other))

		def _backward():
			self.grad += out.grad
			other.grad += out.grad
		out._backward = _backward

		return out

	def __mul__(self, other):
		other = other if isinstance(other, Unity) else Unity(other)
		out = Unity(self.unity * other.unity, (self, other))

		def _backward():
			self.grad += other.unity * out.grad
			other.grad += self.unity * out.grad
		out._backward = _backward

		return out

	def __pow__(self, other):
		if isinstance(other, (int, float)):
			# x ** c
			out = Unity(self.unity ** other, (self,))

			def _backward():
				if self.unity != 0:  # viter NaN
					self.grad += (other * (self.unity ** (other - 1))) * out.grad
			out._backward = _backward

			return out

		elif isinstance(other, Unity):
			# a ** b
			out = Unity(self.unity ** other.unity, (self, other))

			def _backward():
				if self.unity != 0:
					# dy/da = b * a^(b-1)
					self.grad += (other.unity * (self.unity ** (other.unity - 1))) * out.grad
				# dy/db = a^b * ln(a)
				if self.unity > 0:  # ln dfini
					other.grad += (out.unity * math.log(self.unity)) * out.grad
			out._backward = _backward

			return out

		else:
			raise TypeError(f"Unsupported type for power: {type(other)}")


	def relu(self):
		out = Unity(0 if self.unity < 0 else self.unity, (self,))

		def _backward():
			self.grad += (out.unity > 0) * out.grad
		out._backward = _backward

		return out

	def sigmoid(self):
		s = 1 / (1 + math.exp(-self.unity))
		out = Unity(s, (self,))

		def _backward():
			# drive du sigmoid : s * (1 - s)
			self.grad += s * (1 - s) * out.grad

		out._backward = _backward
		return out

	def backward(self):

		# topological order all of the children in the graph
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		build_topo(self)

		# go one variable at a time and apply the chain rule to get its gradient
		self.grad = 1
		for v in reversed(topo):
			v._backward()

	def __neg__(self): # -self
		return self * -1

	def __radd__(self, other): # other + self
		return self + other

	def __sub__(self, other): # self - other
		return self + (-other)

	def __rsub__(self, other): # other - self
		return other + (-self)

	def __rmul__(self, other): # other * self
		return self * other

	def __truediv__(self, other): # self / other
		return self * other**-1

	def __rtruediv__(self, other): # other / self
		return other * self**-1

	def __repr__(self):
		return f"Unity(unity={self.unity}, grad={self.grad})"

# --- IA
class Module:

	def zero_grad(self):
		for p in self.parameters():
			p.grad = 0

	def parameters(self):
		return []


class Node(Module):
	def __init__(self, nin, nonlin=True):
		self.w = [Unity(random.uniform(-1,1)) for _ in range(nin)]
		self.b = Unity(0)
		self.nonlin = nonlin

	def __call__(self, x):
		act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
		if self.nonlin == 0:return act
		if self.nonlin == 1:return act.relu()
		if self.nonlin == 2:return act.sigmoid()

	def parameters(self):
		return self.w + [self.b]

	def __repr__(self):
		return f"{'ReLU' if self.nonlin else 'Linear'}Node({len(self.w)})"

class Layer(Module):

	def __init__(self, nin, nout, **kwargs):
		self.nodes = [Node(nin, **kwargs) for _ in range(nout)]

	def __call__(self, x):
		out = [n(x) for n in self.nodes]
		return out[0] if len(out) == 1 else out

	def parameters(self):
		return [p for n in self.nodes for p in n.parameters()]

	def __repr__(self):
		return f"Layer of [{', '.join(str(n) for n in self.nodes)}]"

class NN(Module):

	def __init__(self, nin, nouts, nonlin=[]):
		sz = [nin] + nouts
		self.layers = [Layer(sz[i], sz[i+1], nonlin=nonlin[i]) for i in range(len(nouts))]

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters()]

	def __repr__(self):
		return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"




def encode_weights_to_unicode_string(weights, offset=12.0, divider=2048.0):
	print("min max:", weights.min(), weights.max())
	weights = np.clip(weights, -offset, offset)
	s = np.round(divider * (weights + offset)).astype(np.uint16)
	bytes_be = bytearray()
	for val in s.flatten():
		bytes_be.append((val >> 8) & 0xFF)
		bytes_be.append(val & 0xFF)
	return bytes_be.decode('utf-16-be')

def decode_unicode_string_to_weights(unicode_str, offset=12.0, divider=2048.0, shape=None):
	# Étape 1 : reconstruire la chaîne binaire 'weights_bytes' comme en C++ wstring -> string
	weights_bytes = bytearray()
	for c in unicode_str:
		val = ord(c)
		weights_bytes.append((val >> 8) & 0xFF)  # octet haut
		weights_bytes.append(val & 0xFF)         # octet bas

	# Étape 2 : lire les poids 2 octets par 2 octets, big-endian
	size = len(weights_bytes) // 2
	output = []
	for i in range(size):
		s1 = weights_bytes[2*i]
		s2 = weights_bytes[2*i + 1]
		s = (s1 << 8) + s2
		val = (s / divider) - offset
		output.append(val)

	# Étape 3 : si shape précisé, reshape en numpy array
	if shape is not None:
		import numpy as np
		output = np.array(output, dtype=np.float32).reshape(shape)
	else:
		output = list(output)

	return output

def export_unity_weights_by_layer(net, output_py="unity_weights.py"):
	"""
	Exporte chaque Layer : W (nout x nin) et b (nout,)
	"""
	with open(output_py, "w", encoding="utf-8") as f:
		f.write("# Unity NN Weights encoded as UTF-16BE unicode strings\n\n")
		
		for layer_idx, layer in enumerate(net.layers):
			# Construire la matrice W : nout x nin
			W = np.array([[wi.unity for wi in node.w] for node in layer.nodes], dtype=np.float32)
			W_str = encode_weights_to_unicode_string(W)
			f.write(f"layer{layer_idx}_W_shape = {W.shape}\n")
			f.write(f"layer{layer_idx}_W = '''{W_str}'''\n\n")
			
			# Construire le vecteur b : nout
			b = np.array([node.b.unity for node in layer.nodes], dtype=np.float32)
			b_str = encode_weights_to_unicode_string(b)
			f.write(f"layer{layer_idx}_b_shape = {b.shape}\n")
			f.write(f"layer{layer_idx}_b = '''{b_str}'''\n\n")
	
	print(f"[o] Poids exportés par layer dans {output_py}")



def import_unity_weights_by_layer(net, module_py):
	import importlib
	uw = importlib.import_module(module_py.replace(".py",""))

	for layer_idx, layer in enumerate(net.layers):
		# Récupérer W et b
		W_str = getattr(uw, f"layer{layer_idx}_W")
		W_shape = getattr(uw, f"layer{layer_idx}_W_shape")
		W = decode_unicode_string_to_weights(W_str, shape=W_shape)
		
		b_str = getattr(uw, f"layer{layer_idx}_b")
		b_shape = getattr(uw, f"layer{layer_idx}_b_shape")
		b = decode_unicode_string_to_weights(b_str, shape=b_shape)

		# Réinjecter dans chaque Node
		for node_idx, node in enumerate(layer.nodes):
			for wi_idx, wi in enumerate(node.w):
				wi.unity = float(W[node_idx, wi_idx])
			node.b.unity = float(b[node_idx])
	
	print("[o] Poids réinjectés par layer dans le modèle Unity")


"""
# Suppose que tu as ton réseau
net = NN(2, [4,1], [1,2])   # exemple

# Export des poids
export_unity_weights_to_unicode(net, "unity_weights.py")

# Plus tard : recharge
import_unicode_weights_to_unity(net, "unity_weights.py")

"""
