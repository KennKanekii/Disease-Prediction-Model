# Import necessary libraries
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('decision_tree_model.joblib')
symptom_list  = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']

mapping = {
    0: 'Fungal_infection',
    1: 'Allergy',
    2: 'GERD',
    3: 'Chronic_cholestasis',
    4: 'Drug_Reaction',
    5: 'Peptic_ulcer_diseae',
    6: 'AIDS',
    7: 'Diabetes',
    8: 'Gastroenteritis',
    9: 'Bronchial_Asthma',
    10: 'Hypertension',
    11: 'Migraine',
    12: 'Cervical_spondylosis',
    13: 'Paralysis_(brain_hemorrhage)',
    14: 'Jaundice',
    15: 'Malaria',
    16: 'Chicken_pox',
    17: 'Dengue',
    18: 'Typhoid',
    19: 'hepatitis_A',
    20: 'Hepatitis_B',
    21: 'Hepatitis_C',
    22: 'Hepatitis_D',
    23: 'Hepatitis_E',
    24: 'Alcoholic_hepatitis',
    25: 'Tuberculosis',
    26: 'Common_Cold',
    27: 'Pneumonia',
    28: 'Dimorphic_hemorrhoids(piles)',
    29: 'Heart_attack',
    30: 'Varicose_veins',
    31: 'Hypothyroidism',
    32: 'Hyperthyroidism',
    33: 'Hypoglycemia',
    34: 'Osteoarthritis',
    35: 'Arthritis',
    36: '(vertigo)_Paroxysmal_Positional_Vertigo',
    37: 'Acne',
    38: 'Urinary_tract_infection',
    39: 'Psoriasis',
    40: 'Impetigo'
}



# Define route for predicting the disease based on symptoms
@app.route('/predict', methods=['POST'])
def predict():
    # Get symptoms from the POST request
    data = request.json
    symptoms = data.get("symptoms", [])  # Expected to be a list of symptom names

    print(type(symptoms))
    print(symptoms)
    # Initialize input vector for the model
    symptom_vector = [0] * len(symptom_list)
    
    for k in range(0,len(symptom_list)):
        for z in symptoms:
            if(z==symptom_list[k]):
                symptom_vector[k]=1

    input_array = [symptom_vector]
    
    # Predict using the model
    prediction = model.predict(input_array)
    predicted_disease = prediction[0]  # Retrieve the predicted disease name

    answer = mapping[predicted_disease]
    
    # Return prediction as JSON
    return jsonify({"predicted_disease": answer})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)