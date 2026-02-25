"""
Smoke test for CRN v2 and DPT v2 notebooks.
Tests: imports, model instantiation, forward pass, loss computation.
Does NOT require dataset or GPU.
"""
import json
import sys
import traceback

def extract_code_cells(nb_file):
    nb = json.loads(open(nb_file, encoding='utf-8').read())
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

def test_notebook(name, nb_file, model_cell_idx):
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {name}")
    print(f"{'='*60}")
    
    cells = extract_code_cells(nb_file)
    
    # ---- Test 1: Imports + Config (Cell 0, skip !pip line) ----
    print("\n[1] Imports + Config...")
    cell0 = cells[0]
    # Remove !pip line
    import_lines = [l for l in cell0.split('\n') if not l.strip().startswith('!')]
    import_code = '\n'.join(import_lines)
    
    ns = {}
    try:
        exec(import_code, ns)
        print("    OK — imports and config loaded")
        # Check key config values
        for key in ['SR', 'N_FFT', 'HOP', 'N_FREQ', 'MAX_LEN', 'BATCH', 'EPOCHS', 'LR']:
            if key in ns:
                print(f"    {key} = {ns[key]}")
            else:
                print(f"    WARNING: {key} not found in config!")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return False
    
    # ---- Test 2: Dataset class (Cell 2) — extract class def only, skip instantiation ----
    print("\n[2] Dataset class definition...")
    try:
        cell2_src = cells[2]
        # Stop before "# Quick sanity check" which needs actual files
        cutoff = cell2_src.find('# Quick sanity check')
        if cutoff == -1:
            cutoff = cell2_src.find('_ds = STFTSpeechDataset')
        if cutoff > 0:
            cell2_src = cell2_src[:cutoff]
        exec(cell2_src, ns)
        assert 'STFTSpeechDataset' in ns, "STFTSpeechDataset not defined!"
        print("    OK — STFTSpeechDataset class defined")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return False
    
    # ---- Test 3: Model class (Cell model_cell_idx) ----
    print(f"\n[3] Model class definition (cell {model_cell_idx})...")
    try:
        exec(cells[model_cell_idx], ns)
        print("    OK — model class defined")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return False
    
    # ---- Test 4: Model instantiation + forward pass ----
    print("\n[4] Model instantiation + forward pass...")
    try:
        import torch
        N_FREQ = ns['N_FREQ']  # 257
        
        # Find the model class
        if 'CRN' in name.upper():
            model_cls = ns.get('CRN')
            if model_cls is None:
                # Try to find it
                for k, v in ns.items():
                    if isinstance(v, type) and issubclass(v, torch.nn.Module) and k != 'nn':
                        model_cls = v
                        break
            model = model_cls(n_freq=N_FREQ)
        else:
            model_cls = ns.get('DPTNet') or ns.get('DPTNetSE')
            if model_cls is None:
                for k, v in ns.items():
                    if isinstance(v, type) and issubclass(v, torch.nn.Module) and k not in ('nn',):
                        if 'DPT' in k or 'Net' in k:
                            model_cls = v
                            break
            model = model_cls(n_freq=N_FREQ)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Model: {model_cls.__name__}, params: {n_params:,}")
        
        # Forward pass with dummy data
        # Input: log1p(magnitude) spectrogram [B, 1, F, T] (unsqueeze channel dim)
        B, T = 2, 188  # 188 frames ~ 3s at hop=256, sr=16000
        x = torch.randn(B, 1, N_FREQ, T).abs()  # magnitude is non-negative
        
        model.eval()
        with torch.no_grad():
            out = model(x)
        
        print(f"    Input shape:  {list(x.shape)}")
        print(f"    Output shape: {list(out.shape)}")
        # Model outputs mask [B, F, T] from input [B, 1, F, T]
        expected_out = (B, N_FREQ, T)
        assert tuple(out.shape) == expected_out, f"Shape mismatch! Expected {expected_out} Got {tuple(out.shape)}"
        print("    OK — forward pass output shape correct")
        
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return False
    
    # ---- Test 5: Loss computation ----
    print("\n[5] Loss computation...")
    try:
        import torch
        # Simulate masked STFT loss (mask is [B, F, T], magnitude is [B, F, T])
        noisy_mag = torch.randn(B, N_FREQ, T).abs()
        target_mag = torch.randn(B, N_FREQ, T).abs()
        enhanced_mag = out * noisy_mag  # mask * input magnitude
        eps = 1e-8
        loss = torch.nn.functional.l1_loss(
            torch.log(enhanced_mag + eps), torch.log(target_mag + eps)
        )
        print(f"    L1 loss on log-magnitude: {loss.item():.4f}")
        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is Inf!"
        print("    OK — loss is finite")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return False
    
    # ---- Test 6: SI-SDR utility (Cell 4) ----
    print("\n[6] SI-SDR utility...")
    try:
        exec(cells[4], ns)
        si_sdr_fn = ns.get('si_sdr')
        if si_sdr_fn:
            est = torch.randn(2, 48000)
            ref = torch.randn(2, 48000)
            val = si_sdr_fn(est, ref)
            print(f"    si_sdr output: {val.item():.2f} dB")
            print("    OK")
        else:
            print("    WARNING: si_sdr function not found, skipping")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return False
    
    print(f"\n{'='*60}")
    print(f"SMOKE TEST PASSED: {name}")
    print(f"{'='*60}")
    return True


if __name__ == '__main__':
    ok1 = test_notebook("CRN v2", "crn_v2_nb_text.txt", model_cell_idx=3)
    ok2 = test_notebook("DPT v2", "dpt_v2_nb_text.txt", model_cell_idx=3)
    
    print(f"\n\n{'='*60}")
    print(f"RESULTS: CRN={'PASS' if ok1 else 'FAIL'}  DPT={'PASS' if ok2 else 'FAIL'}")
    print(f"{'='*60}")
    
    sys.exit(0 if (ok1 and ok2) else 1)
