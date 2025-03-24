import os
import zipfile
import tempfile
import numpy as np
import cv2
import glob

from psnr_ssim import calculate_psnr, calculate_ssim

# 수정: Ground truth 이미지들이 저장된 디렉토리 경로 (반드시 실제 경로로 변경)
GROUND_TRUTH_DIR = "/path/to/ground_truth"

# 유효한 이미지 확장자
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.dng')

def build_ground_truth_dict(gt_dir):
    """
    Ground truth 폴더 내의 모든 이미지 파일에 대해,
    파일명에서 공통 식별자(예: "Case113")를 키로 하여 매핑하는 딕셔너리를 생성합니다.
    (예: "Case113_O.png" -> key: "Case113")
    """
    gt_files = []
    for ext in VALID_EXTENSIONS:
        gt_files.extend(glob.glob(os.path.join(gt_dir, f"*{ext}")))
    
    gt_dict = {}
    for file_path in gt_files:
        base = os.path.basename(file_path)
        # 예: "Case113_O.png" 에서 "_"로 분리하여 첫번째 부분을 key로 사용
        if "_" in base:
            key = base.split("_")[0]
            gt_dict[key] = file_path
    return gt_dict

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    output = {}

    # 1. 제출된 zip 파일 압축 해제
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(user_submission_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except Exception as e:
        print("Error extracting zip file:", e)
        output["result"] = [{"train_split": {"Metric1": 0, "Metric2": 0, "Metric3": 0, "Total": 0}}]
        output["submission_result"] = output["result"][0]["train_split"]
        return output

    # 2. 압축 해제된 폴더에서 제출 이미지 파일들 찾기
    submission_files = []
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                submission_files.append(os.path.join(root, f))
    
    if not submission_files:
        print("No valid submission image files found.")
        output["result"] = [{"train_split": {"Metric1": 0, "Metric2": 0, "Metric3": 0, "Total": 0}}]
        output["submission_result"] = output["result"][0]["train_split"]
        return output

    # 3. Ground truth 파일 매핑 딕셔너리 생성
    gt_dict = build_ground_truth_dict(GROUND_TRUTH_DIR)
    if not gt_dict:
        print("No ground truth images found in", GROUND_TRUTH_DIR)
        output["result"] = [{"train_split": {"Metric1": 0, "Metric2": 0, "Metric3": 0, "Total": 0}}]
        output["submission_result"] = output["result"][0]["train_split"]
        return output

    psnr_list = []
    ssim_list = []
    count = 0

    # 4. 각 제출 이미지 파일에 대해 평가 수행
    for sub_file in submission_files:
        base_name = os.path.basename(sub_file)
        # 제출 파일의 공통 식별자 추출 (예: "Case113_D.png" -> "Case113")
        if "_" in base_name:
            common_id = base_name.split("_")[0]
        else:
            # "_"가 없으면 파일명 전체(확장자 제외) 사용
            common_id = os.path.splitext(base_name)[0]

        # Ground truth 파일은 common_id + "_O" 로 되어 있다고 가정 (확장자는 제출 파일과 동일하게 맞춤)
        gt_expected = f"{common_id}_O"  # 확장자는 따로 추가
        # ground truth 폴더 내에서 제출 파일과 동일한 확장자를 가진 파일 중 이름에 gt_expected가 포함된 파일 검색
        gt_candidates = [f for f in os.listdir(GROUND_TRUTH_DIR)
                         if f.lower().endswith(os.path.splitext(base_name)[1].lower()) and gt_expected in f]
        if not gt_candidates:
            print(f"No ground truth file found for submission {base_name} with key {gt_expected}. Skipping.")
            continue
        # 만약 여러 후보가 있다면 첫 번째 파일 사용
        gt_file = os.path.join(GROUND_TRUTH_DIR, gt_candidates[0])
        
        # 이미지 읽기 (cv2.imread는 BGR 순서로 읽음)
        gt_img = cv2.imread(gt_file)
        sub_img = cv2.imread(sub_file)
        if gt_img is None or sub_img is None:
            print(f"Error reading images for {base_name}. Skipping.")
            continue
        
        # 이미지 크기가 다르면, 제출 이미지를 원본 크기로 리사이즈
        if gt_img.shape != sub_img.shape:
            sub_img = cv2.resize(sub_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        crop_border = 0  # 필요 시 조정
        try:
            psnr_val = calculate_psnr(gt_img, sub_img, crop_border, input_order='HWC', test_y_channel=False)
            ssim_val = calculate_ssim(gt_img, sub_img, crop_border, input_order='HWC', test_y_channel=False)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            count += 1
        except Exception as e:
            print(f"Error calculating metrics for {base_name}: {e}")
            continue

    # 5. 평균 PSNR, SSIM 계산
    if count > 0:
        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))
    else:
        avg_psnr = 0.0
        avg_ssim = 0.0

    # 6. 결과 저장: 여기서는 PSNR을 Total 점수로 사용 (높을수록 좋은 경우)
    result_dict = {
        "train_split": {
            "Metric1": avg_ssim,  # SSIM
            "Metric2": avg_psnr,  # PSNR
            "Metric3": 0,
            "Total": avg_psnr
        }
    }
    
    output["result"] = [result_dict]
    output["submission_result"] = result_dict["train_split"]
    print("Completed evaluation.")
    return output
