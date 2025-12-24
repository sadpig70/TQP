"""
BeH₂ CASSCF 계산 스크립트

PySCF를 사용하여 BeH₂ 분자의 CASSCF(2,2) 계산을 수행합니다.

사용법:
    python beh2_casscf.py

출력:
    - 콘솔: HF/CASSCF/FCI 에너지
    - 파일: beh2_integrals.npz (1e/2e 적분)
"""

import numpy as np

def get_beh2_geometry():
    """BeH₂ 선형 분자 기하 구조 반환 (Å 단위)"""
    # Be-H 결합 거리: 1.326 Å (평형 거리)
    # 선형 구조: H-Be-H
    return """
    Be  0.0000  0.0000  0.0000
    H   0.0000  0.0000  1.3260
    H   0.0000  0.0000 -1.3260
    """

def run_hf_calculation(mol):
    """Hartree-Fock 계산 실행"""
    from pyscf import scf
    
    mf = scf.RHF(mol)
    mf.verbose = 4
    e_hf = mf.kernel()
    
    print(f"\n{'='*50}")
    print(f"HF Energy: {e_hf:.6f} Ha")
    print(f"{'='*50}\n")
    
    return mf, e_hf

def run_casscf_calculation(mf, n_electrons=2, n_orbitals=2):
    """CASSCF 계산 실행
    
    Args:
        mf: HF 결과 객체
        n_electrons: 활성 전자 수 (기본: 2)
        n_orbitals: 활성 궤도 수 (기본: 2)
    
    Note:
        BeH₂는 6전자 시스템이므로 활성 전자 수가 짝수여야 함.
        CASSCF(2,2): 2 활성 전자, 2 활성 궤도 (4 코어 전자)
    """
    from pyscf import mcscf
    
    # CASSCF(2,2): 2 전자, 2 궤도 (core: 4 전자)
    mc = mcscf.CASSCF(mf, n_orbitals, n_electrons)
    mc.verbose = 4
    e_casscf = mc.kernel()[0]
    
    print(f"\n{'='*50}")
    print(f"CASSCF({n_electrons},{n_orbitals}) Energy: {e_casscf:.6f} Ha")
    print(f"{'='*50}\n")
    
    return mc, e_casscf

def run_fci_calculation(mf):
    """FCI 계산 실행 (참조용)"""
    from pyscf import fci
    
    # 전체 FCI (큰 시스템에서는 비용이 높음)
    cisolver = fci.FCI(mf)
    e_fci = cisolver.kernel()[0]
    
    print(f"\n{'='*50}")
    print(f"FCI Energy: {e_fci:.6f} Ha")
    print(f"{'='*50}\n")
    
    return e_fci

def extract_integrals(mc):
    """CASSCF 결과에서 1-body/2-body 적분 추출"""
    # 1-전자 적분 (활성 공간)
    h1e_result = mc.get_h1cas()
    h1e = h1e_result[0] if isinstance(h1e_result, tuple) else h1e_result
    
    # 2-전자 적분 (활성 공간)
    h2e_result = mc.get_h2cas()
    h2e = h2e_result[0] if isinstance(h2e_result, tuple) else h2e_result
    
    # 핵간 반발 에너지
    e_nuc = mc.mol.energy_nuc()
    
    # 코어 에너지 계산 (PySCF 버전별 호환성)
    try:
        e_core = mc.energy_core
    except AttributeError:
        # CASSCF 에너지 분해로 계산: E_tot = E_core + E_cas + E_nuc
        e_cas = mc.e_cas if hasattr(mc, 'e_cas') else 0.0
        e_core = mc.e_tot - e_cas - e_nuc
    
    print(f"\n{'='*50}")
    print(f"Extracted Integrals:")
    print(f"  h1e shape: {h1e.shape}")
    print(f"  h2e shape: {h2e.shape}")
    print(f"  Nuclear repulsion: {e_nuc:.6f} Ha")
    print(f"  Core energy: {e_core:.6f} Ha")
    print(f"{'='*50}\n")
    
    return h1e, h2e, e_nuc, e_core

def save_integrals(h1e, h2e, e_nuc, e_core, filename="beh2_integrals.npz"):
    """적분을 파일로 저장"""
    np.savez(
        filename,
        h1e=h1e,
        h2e=h2e,
        e_nuc=e_nuc,
        e_core=e_core
    )
    print(f"Integrals saved to {filename}")

def main():
    """메인 실행 함수"""
    try:
        from pyscf import gto
    except ImportError:
        print("Error: PySCF가 설치되지 않았습니다.")
        print("설치: pip install pyscf")
        return
    
    print("="*60)
    print("BeH₂ CASSCF Calculation")
    print("="*60)
    
    # 1. 분자 정의
    mol = gto.Mole()
    mol.atom = get_beh2_geometry()
    mol.basis = 'sto-3g'
    mol.charge = 0
    mol.spin = 0  # 단일항
    mol.build()
    
    print(f"\nMolecule: BeH₂")
    print(f"Basis: {mol.basis}")
    print(f"Electrons: {mol.nelectron}")
    print(f"Orbitals: {mol.nao}")
    
    # 2. HF 계산
    mf, e_hf = run_hf_calculation(mol)
    
    # 3. CASSCF 계산 (2 활성 전자, 2 활성 궤도)
    mc, e_casscf = run_casscf_calculation(mf, n_electrons=2, n_orbitals=2)
    
    # 4. FCI 계산 (참조)
    try:
        e_fci = run_fci_calculation(mf)
    except Exception as e:
        print(f"FCI calculation skipped: {e}")
        e_fci = None
    
    # 5. 적분 추출
    h1e, h2e, e_nuc, e_core = extract_integrals(mc)
    
    # 6. 적분 저장
    save_integrals(h1e, h2e, e_nuc, e_core)
    
    # 7. 결과 요약
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"E_HF:     {e_hf:.6f} Ha")
    print(f"E_CASSCF: {e_casscf:.6f} Ha")
    if e_fci:
        print(f"E_FCI:    {e_fci:.6f} Ha")
        print(f"Correlation (CASSCF): {(e_casscf - e_hf)*1000:.2f} mHa")
        print(f"Correlation (FCI):    {(e_fci - e_hf)*1000:.2f} mHa")
    print("="*60)

if __name__ == "__main__":
    main()
