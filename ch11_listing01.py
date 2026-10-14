from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Grant:
    tenant: str
    tools: frozenset
    document_ids: frozenset

def authorize(grant, tenant, tool, document_id):
    return (tenant == grant.tenant
            and tool in grant.tools
            and document_id in grant.document_ids)

grant = Grant('A', frozenset({'read'}),
              frozenset({'doc-A'}))
tests = [('A', 'read', 'doc-A', True),
         ('B', 'read', 'doc-A', False),
         ('A', 'write', 'doc-A', False),
         ('A', 'read', 'doc-B', False)]
for tenant, tool, doc, expected in tests:
    assert authorize(grant, tenant, tool, doc) == expected

def wilson(k, n, z=1.96):
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    radius /= 1 + z*z/n
    return center-radius, center+radius

print('authorization: 4/4')
print('zero failures among 100:', wilson(0, 100))
print('zero failures among 1000:', wilson(0, 1000))
