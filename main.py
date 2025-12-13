# Importation des bibliothèques
import pandas as pd
from landmark_extraction import pipe_extract_landmark
from grader import compute_repgrade
from display import play_video_with_landmarks_and_reps
from classifier import predict_exercise

def main():
    """
    Pipeline d'exécution complète pour une vidéo de musculation.
    """
# ==============================================================================
# Partie 1 : Lecture de la vidéo et extraction des landmarks
# ==============================================================================

    filename ="data/raw_data/data-eval/push-up/push-up_5.mp4"
    landmarks, df = pipe_extract_landmark(filename)
    
# ==============================================================================
# Partie 2 : Classification de la vidéo
# ==============================================================================
    
    predicted_exercise = predict_exercise(df)
    print(f"Exercice prédit : {predicted_exercise}")
    
# ==============================================================================
# Partie 3 : Comptage des répétitions et calcul des scores
# ==============================================================================

    rep_starts, sim_list = compute_repgrade(df, predicted_exercise)
    print(rep_starts)
    print(sim_list)
    play_video_with_landmarks_and_reps(filename, landmarks, rep_starts, sim_list)
    
if __name__ == "__main__":
    main()