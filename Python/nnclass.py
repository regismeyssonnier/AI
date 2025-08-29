# -*- coding: latin-1 -*-

import contextlib
import exportnn as expnn
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
			self._prev = set()         # pas de d�pendances
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

	def tanh(self):
		t = math.tanh(self.unity)      # valeur de tanh(x)
		out = Unity(t, (self,))

		def _backward():
			# d�riv�e : 1 - tanh(x)^2
			self.grad += (1 - t**2) * out.grad

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
		if self.nonlin == 3:return act.tanh()

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

