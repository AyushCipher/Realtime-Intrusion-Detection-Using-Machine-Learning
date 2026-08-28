"""A small, curated MITRE ATT&CK (Enterprise) knowledge base for grounding
Tier 2's LLM explanations via retrieval (RAG), instead of asking the model
to name techniques from memory alone.

**This is not the full ATT&CK corpus.** It's a handful of technique
entries chosen to cover the attack families this project's own
`ATTACK_CATEGORY_MAP` (see `ids_ml/src/ids_ml/data.py`) models: DoS/DDoS,
Brute Force, Web Attack, Infiltration, Botnet, PortScan. Building or
fetching the full ~200-technique ATT&CK STIX corpus was out of scope for
this pass; extending this dict with more entries (or replacing it with a
loader over the real ATT&CK STIX bundle) is the natural next step -- the
retrieval mechanism in `retrieval.py` doesn't care how many entries there
are.

**The category -> technique mapping here is this project's own reasonable
association, not a verified ground truth.** CICIDS2017/2018 (the datasets
`ml/` trains and evaluates on) don't ship with ATT&CK technique labels --
they're pre-ATT&CK-adoption academic datasets with coarse attack-family
labels ("DoS Hulk", "FTP-BruteForce", etc.), not technique IDs. The
mappings below associate each family with the ATT&CK technique(s) most
directly descriptive of that traffic pattern, to the author's best
knowledge of the framework, but should be read as illustrative grounding
for the LLM's explanation, not as a citation-grade dataset annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

Category = str  # matches ids_ml.pipeline's stage2_predicted_class / attack_category vocabulary


@dataclass(frozen=True)
class TechniqueEntry:
    technique_id: str
    name: str
    tactic: str
    category: Category  # which of this project's attack families this technique grounds
    description: str


KNOWLEDGE_BASE: List[TechniqueEntry] = [
    TechniqueEntry(
        technique_id="T1498",
        name="Network Denial of Service",
        tactic="Impact",
        category="DoS/DDoS",
        description=(
            "Adversaries may perform network denial-of-service (DoS) attacks to degrade or block the "
            "availability of targeted resources, by exhausting network bandwidth or connection-handling "
            "capacity through a flood of malicious traffic. Direct floods send high-volume traffic from "
            "a single or several sources; reflection/amplification attacks abuse intermediary servers to "
            "multiply attacker-controlled traffic before it reaches the target."
        ),
    ),
    TechniqueEntry(
        technique_id="T1499",
        name="Endpoint Denial of Service",
        tactic="Impact",
        category="DoS/DDoS",
        description=(
            "Adversaries may target the various components of a network service to degrade or block its "
            "availability, e.g. by exhausting a web server's request-handling threads or memory with a "
            "large volume of slow or malformed application-layer requests (application exhaustion flood), "
            "rather than saturating raw network bandwidth."
        ),
    ),
    TechniqueEntry(
        technique_id="T1110",
        name="Brute Force",
        tactic="Credential Access",
        category="Brute Force",
        description=(
            "Adversaries may use brute force techniques to gain access to accounts when passwords are "
            "unknown or when password hashes are obtained. Password guessing repeatedly attempts likely "
            "passwords for one or a few known accounts (e.g. against SSH or FTP); password spraying "
            "attempts one or a few common passwords across many accounts to avoid account lockouts."
        ),
    ),
    TechniqueEntry(
        technique_id="T1190",
        name="Exploit Public-Facing Application",
        tactic="Initial Access",
        category="Web Attack",
        description=(
            "Adversaries may attempt to exploit a weakness in an Internet-facing host or system to gain "
            "initial access, using software, data, or commands to cause unintended or unanticipated "
            "behavior. This includes exploitation of bugs in web applications, such as SQL injection "
            "(unsanitized input reaching a database query) or cross-site scripting (unsanitized input "
            "reflected back into a page as executable script)."
        ),
    ),
    TechniqueEntry(
        technique_id="T1046",
        name="Network Service Discovery",
        tactic="Discovery",
        category="PortScan",
        description=(
            "Adversaries may attempt to get a listing of services running on remote hosts, including those "
            "that may be vulnerable to remote software exploitation, by systematically probing ports and "
            "protocols across a target range -- a port scan. This is frequently a reconnaissance step "
            "preceding a more targeted intrusion attempt."
        ),
    ),
    TechniqueEntry(
        technique_id="T1210",
        name="Exploitation of Remote Services",
        tactic="Lateral Movement",
        category="Infiltration",
        description=(
            "Adversaries may exploit remote services to gain unauthorized access to internal systems once "
            "initial access is achieved, often following a reconnaissance phase. In the CICIDS2018 "
            "infiltration scenario specifically, a client-side exploit (a malicious document) delivers "
            "initial access, followed by internal network service discovery and lateral movement -- so "
            "infiltration traffic often co-occurs with discovery-like scanning patterns from the same "
            "internal host shortly after the initial exploit."
        ),
    ),
    TechniqueEntry(
        technique_id="T1071.001",
        name="Application Layer Protocol: Web Protocols",
        tactic="Command and Control",
        category="Botnet",
        description=(
            "Adversaries may communicate using application-layer protocols (commonly HTTP/HTTPS) to avoid "
            "detection by blending in with existing traffic, to relay commands from a command-and-control "
            "(C2) server to compromised hosts (bots) and receive results back. Botnet traffic often shows "
            "periodic, low-volume beaconing rather than the burst patterns typical of DoS or brute-force "
            "traffic."
        ),
    ),
]


def categories_covered() -> List[str]:
    return sorted({entry.category for entry in KNOWLEDGE_BASE})
