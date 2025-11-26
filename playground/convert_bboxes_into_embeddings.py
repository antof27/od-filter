import torch
import os
import glob
from tqdm import tqdm

def concatenate_and_clean(input_folder, output_folder):
    # 1. Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)
    
    pt_files = glob.glob(os.path.join(input_folder, "*.pt"))
    print(f"Found {len(pt_files)} files in {input_folder}")

    for f_path in tqdm(pt_files, desc="Concatenating Embeddings"):
        try:
            # Carica il file
            data = torch.load(f_path, map_location='cpu')
            
            # Estrai i tensori
            embeddings = data['embeddings'] # Shape: [N, 1280]
            bboxes = data['bboxes']         # Shape: [N, 4] (Già normalizzate)
            
            # Controllo di sicurezza sulle dimensioni
            if embeddings.shape[0] != bboxes.shape[0]:
                print(f"[Error] Mismatch in {f_path}: {embeddings.shape[0]} embeddings vs {bboxes.shape[0]} bboxes. Skipping.")
                continue
            
            # --- 2. CONCATENAZIONE ---
            # Uniamo lungo la dimensione 1 (quella delle features)
            # 1280 + 4 = 1284
            # Assicuriamoci che bboxes sia float32 come gli embeddings
            new_embeddings = torch.cat([embeddings.float(), bboxes.float()], dim=1)
            
            # --- 3. AGGIORNAMENTO DIZIONARIO ---
            # Sovrascriviamo gli embeddings con quelli nuovi
            data['embeddings'] = new_embeddings
            
            # Rimuoviamo la chiave bboxes per risparmiare spazio e pulire
            if 'bboxes' in data:
                del data['bboxes']
                
            # (Opzionale) Rimuovi anche width/height se erano stati salvati e non servono più
            # if 'width' in data: del data['width']
            # if 'height' in data: del data['height']

            # --- 4. SALVATAGGIO ---
            filename = os.path.basename(f_path)
            save_path = os.path.join(output_folder, filename)
            torch.save(data, save_path)

        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    print(f"Done! New files with shape [N, 1284] saved in: {output_folder}")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    
    # Cartella contenente i file con le box già normalizzate (output dello step precedente)
    INPUT_FOLDER = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concat-embeds-bboxes-norm" 
    
    # Nuova cartella finale pronta per il training
    OUTPUT_FOLDER = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concat-embeds-1284"
    
    concatenate_and_clean(INPUT_FOLDER, OUTPUT_FOLDER)