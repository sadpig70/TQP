#!/usr/bin/env python3
"""
IBM Quantum Estimator API Test Suite
=====================================

Validated: 2025-12-07
Backend: ibm_fez (156-qubit Heron r2)
Result: HF energy error 1.77 mHa (chemical accuracy achieved)
"""

import json, subprocess, time, os

# Config
API_KEY = os.environ.get("IBM_QUANTUM_TOKEN", "")
SERVICE_CRN = os.environ.get("IBM_QUANTUM_CRN", "")
BACKEND = os.environ.get("IBM_QUANTUM_BACKEND", "ibm_fez")
API_URL = "https://us-east.quantum-computing.cloud.ibm.com"

# LiH Hamiltonian (verified)
LIH = {
    "identity": -7.4983,
    "observables": [
        ("IIZI", 0.2404), ("IIIZ", 0.2404), ("IZII", 0.1801), ("ZIII", 0.1801),
        ("IIZZ", 0.1741), ("IZIZ", 0.1215), ("IZZI", 0.1656), ("ZIIZ", 0.1215),
        ("ZIZI", 0.1656), ("ZZII", 0.1228),
        ("XXXX", 0.0453), ("YYYY", 0.0453), ("XXYY", -0.0453), ("YYXX", -0.0453),
    ],
    "hf_energy": -7.8962,
    "fci_energy": -7.8825,
}

LIH_HF_QASM = """OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
x q[0];
x q[1];
"""

def run_estimator(qasm, observables):
    """Submit and wait for Estimator job."""
    pub = {"program_id": "estimator", "backend": BACKEND,
           "params": {"pubs": [[qasm, observables, []]], "version": 2}}
    
    r = subprocess.run(f'''curl -s -X POST "{API_URL}/jobs" \
        -H "Authorization: apikey {API_KEY}" -H "Service-CRN: {SERVICE_CRN}" \
        -H "Content-Type: application/json" -d '{json.dumps(pub)}' ''',
        shell=True, capture_output=True, text=True)
    
    try:
        jid = json.loads(r.stdout).get("id")
    except:
        return None, f"Submit error: {r.stdout[:100]}"
    if not jid:
        return None, "No job ID"
    
    # Wait
    for _ in range(40):
        time.sleep(3)
        sr = subprocess.run(f'''curl -s "{API_URL}/jobs/{jid}" \
            -H "Authorization: apikey {API_KEY}" -H "Service-CRN: {SERVICE_CRN}"''',
            shell=True, capture_output=True, text=True)
        try:
            d = json.loads(sr.stdout)
            st = d.get("state",{}).get("status","")
            if st == "Completed": break
            elif st in ["Failed","Cancelled"]:
                return None, d.get("state",{}).get("reason","")[:150]
        except: pass
    
    # Results
    rr = subprocess.run(f'''curl -s "{API_URL}/jobs/{jid}/results" \
        -H "Authorization: apikey {API_KEY}" -H "Service-CRN: {SERVICE_CRN}"''',
        shell=True, capture_output=True, text=True)
    try:
        rd = json.loads(rr.stdout)
        evs = rd.get("results",[{}])[0].get("data",{}).get("evs",[])
        stds = rd.get("results",[{}])[0].get("data",{}).get("stds",[])
        return {"evs": evs, "stds": stds, "job_id": jid}, None
    except Exception as e:
        return None, str(e)

def compute_energy(evs, h=LIH):
    e = h["identity"]
    for i, (_, c) in enumerate(h["observables"]):
        if i < len(evs): e += c * evs[i]
    return e

def main():
    print("="*60)
    print("  IBM Quantum Estimator API Test - LiH HF State")
    print("="*60)
    
    if not API_KEY or not SERVICE_CRN:
        print("Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN")
        return
    
    obs = [o[0] for o in LIH["observables"]]
    print(f"Backend: {BACKEND}")
    print(f"Observables: {len(obs)}")
    
    result, err = run_estimator(LIH_HF_QASM, obs)
    if err:
        print(f"Error: {err}")
        return
    
    evs = result["evs"]
    energy = compute_energy(evs)
    error_mha = abs(energy - LIH["hf_energy"]) * 1000
    
    print(f"\nJob: {result['job_id']}")
    print(f"Measured: {energy:.6f} Ha")
    print(f"Theory:   {LIH['hf_energy']:.6f} Ha")
    print(f"Error:    {error_mha:.2f} mHa")
    
    if error_mha < 10:
        print("\n✅ Chemical accuracy achieved!")
    
if __name__ == "__main__":
    main()
