import face_recognition
import numpy as np
import cv2
import argparse
import os
import glob
from typing import Tuple, List, Dict, Optional

class FaceMatcher:
    def __init__(self, tolerance: float = 0.6, high_confidence_threshold: float = 0.75):
        """
        Initialize the FaceMatcher with a tolerance threshold.
        
        Args:
            tolerance: The threshold for face recognition. Lower values are more strict.
                       Recommended values: 0.6 (default), 0.5 (stricter) or 0.7 (more lenient)
            high_confidence_threshold: Threshold for considering a match as high confidence (0.75 = 75%)
        """
        self.tolerance = tolerance
        self.high_confidence_threshold = high_confidence_threshold
        self.high_confidence_matches = []  # To store high confidence matches
    
    def load_image(self, image_path: str):
        """Load an image from a file path and convert to RGB."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image using OpenCV and convert from BGR to RGB
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def find_faces(self, image, image_path=None) -> Tuple[List, List]:
        """
        Find all faces in an image.
        
        Returns:
            Tuple containing face locations and face encodings
        """
        # Find face locations in the image
        face_locations = face_recognition.face_locations(image)
        
        # Get face encodings for each face found
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if image_path and not face_locations:
            print(f"No faces found in: {image_path}")
        
        return face_locations, face_encodings
    
    def match_faces_across_folder(self, query_folder: str, dataset_folder: str, 
                               output_folder: str = "results", 
                               image_extensions: List[str] = None) -> Dict:
        """
        Match faces from query folder against all images in dataset folder.
        
        Args:
            query_folder: Path to folder containing query images
            dataset_folder: Path to folder containing dataset images to search through
            output_folder: Path to folder where results will be saved
            image_extensions: List of image extensions to process (default: ['.jpg', '.jpeg', '.png'])
            
        Returns:
            Dictionary with query image paths as keys and lists of matches as values
        """
        # Reset high confidence matches list for this run
        self.high_confidence_matches = []
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png']
            
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image paths in both folders
        query_paths = []
        for ext in image_extensions:
            query_paths.extend(glob.glob(os.path.join(query_folder, f"*{ext}")))
            query_paths.extend(glob.glob(os.path.join(query_folder, f"*{ext.upper()}")))
            
        dataset_paths = []
        for ext in image_extensions:
            dataset_paths.extend(glob.glob(os.path.join(dataset_folder, f"*{ext}")))
            dataset_paths.extend(glob.glob(os.path.join(dataset_folder, f"*{ext.upper()}")))
            
        if not query_paths:
            raise ValueError(f"No images found in query folder: {query_folder}")
            
        if not dataset_paths:
            raise ValueError(f"No images found in dataset folder: {dataset_folder}")
            
        print(f"Found {len(query_paths)} query images and {len(dataset_paths)} dataset images")
        
        # Process each query image
        all_matches = {}
        
        for query_idx, query_path in enumerate(query_paths):
            query_filename = os.path.basename(query_path)
            print(f"Processing query image {query_idx+1}/{len(query_paths)}: {query_filename}")
            
            # Load query image and find faces
            try:
                query_image = self.load_image(query_path)
                query_face_locations, query_face_encodings = self.find_faces(query_image, query_path)
                
                if not query_face_encodings:
                    print(f"  No faces found in query image: {query_filename}")
                    all_matches[query_path] = []
                    continue
                
                # Process each face in the query image
                query_matches = []
                
                for face_idx, query_encoding in enumerate(query_face_encodings):
                    print(f"  Searching for face {face_idx+1}/{len(query_face_encodings)} from {query_filename}")
                    face_matches = []
                    
                    # Compare against all dataset images
                    for dataset_idx, dataset_path in enumerate(dataset_paths):
                        # Skip if comparing to the same image
                        if os.path.abspath(dataset_path) == os.path.abspath(query_path):
                            continue
                            
                        try:
                            dataset_image = self.load_image(dataset_path)
                            dataset_face_locations, dataset_face_encodings = self.find_faces(dataset_image)
                            
                            # If faces found, compare with query face
                            for dataset_face_idx, dataset_encoding in enumerate(dataset_face_encodings):
                                # Calculate face distance (lower = more similar)
                                face_distance = face_recognition.face_distance([dataset_encoding], query_encoding)[0]
                                similarity = 1 - face_distance  # Convert distance to similarity score
                                
                                # Check if the faces match within tolerance
                                if face_distance <= self.tolerance:
                                    match_info = {
                                        'query_face_idx': face_idx,
                                        'query_path': query_path,
                                        'query_filename': query_filename,
                                        'dataset_path': dataset_path,
                                        'dataset_filename': os.path.basename(dataset_path),
                                        'dataset_face_idx': dataset_face_idx,
                                        'similarity': similarity,
                                        'dataset_face_location': dataset_face_locations[dataset_face_idx],
                                    }
                                    face_matches.append(match_info)
                                    
                                    # Check for high confidence matches
                                    if similarity >= self.high_confidence_threshold:
                                        self.high_confidence_matches.append(match_info)
                        
                        except Exception as e:
                            print(f"  Error processing dataset image {dataset_path}: {e}")
                    
                    # Sort matches by similarity (highest first)
                    face_matches.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Add to query matches
                    query_matches.append({
                        'query_face_idx': face_idx,
                        'query_face_location': query_face_locations[face_idx],
                        'matches': face_matches
                    })
                    
                    print(f"  Found {len(face_matches)} matches for face {face_idx+1}")
                
                # Store all matches for this query image
                all_matches[query_path] = query_matches
                
                # Create visualization for this query
                if query_matches:
                    self._create_match_visualization(query_path, query_image, query_matches, output_folder)
            
            except Exception as e:
                print(f"Error processing query image {query_path}: {e}")
                all_matches[query_path] = []
        
        # Generate a summary report
        self._generate_summary_report(all_matches, output_folder)
        
        # Generate a high confidence report
        if self.high_confidence_matches:
            self._generate_high_confidence_report(output_folder)
        
        return all_matches
    
    def _create_match_visualization(self, query_path, query_image, query_matches, output_folder):
        """Create a visualization of matches for a query image."""
        query_filename = os.path.basename(query_path)
        query_name = os.path.splitext(query_filename)[0]
        
        # Create a folder for this query
        query_output_folder = os.path.join(output_folder, query_name)
        os.makedirs(query_output_folder, exist_ok=True)
        
        # For each face in the query image
        for face_match in query_matches:
            query_face_idx = face_match['query_face_idx']
            face_matches = face_match['matches']
            
            if not face_matches:
                continue
                
            # Only visualize top 5 matches to avoid cluttering
            top_matches = face_matches[:5]
            
            # Get face location in query image
            top, right, bottom, left = face_match['query_face_location']
            
            # Create a copy of the query image with face highlighted
            query_display = cv2.cvtColor(query_image.copy(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(query_display, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add face number
            cv2.putText(query_display, f"Face #{query_face_idx+1}", 
                       (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create a grid of matches
            grid_images = [query_display]
            
            for match_idx, match in enumerate(top_matches):
                try:
                    # Load dataset image
                    dataset_image = self.load_image(match['dataset_path'])
                    dataset_filename = os.path.basename(match['dataset_path'])
                    
                    # Highlight matched face
                    dataset_display = cv2.cvtColor(dataset_image.copy(), cv2.COLOR_RGB2BGR)
                    m_top, m_right, m_bottom, m_left = match['dataset_face_location']
                    
                    # Use different color for high confidence matches
                    color = (0, 0, 255) if match['similarity'] >= self.high_confidence_threshold else (0, 255, 0)
                    cv2.rectangle(dataset_display, (m_left, m_top), (m_right, m_bottom), color, 2)
                    
                    # Add match info
                    confidence_pct = int(match['similarity'] * 100)
                    similarity_text = f"Match #{match_idx+1}: {confidence_pct}%"
                    cv2.putText(dataset_display, similarity_text, 
                               (m_left, m_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(dataset_display, dataset_filename, 
                               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    grid_images.append(dataset_display)
                    
                except Exception as e:
                    print(f"  Error creating visualization for match: {e}")
            
            # Create a grid of images (1 row with query + matches)
            if grid_images:
                # Resize all images to the same height
                max_height = 300
                resized_images = []
                
                for img in grid_images:
                    h, w = img.shape[:2]
                    new_width = int(w * (max_height / h))
                    resized = cv2.resize(img, (new_width, max_height))
                    resized_images.append(resized)
                
                # Concatenate horizontally
                final_image = np.hstack(resized_images)
                
                # Save the result
                result_filename = f"{query_name}_face{query_face_idx+1}_matches.jpg"
                result_path = os.path.join(query_output_folder, result_filename)
                cv2.imwrite(result_path, final_image)
                print(f"  Saved match visualization to {result_path}")
    
    def _generate_summary_report(self, all_matches, output_folder):
        """Generate a text report summarizing all matches."""
        report_path = os.path.join(output_folder, "matching_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("Face Matching Summary Report\n")
            f.write("==========================\n\n")
            
            for query_path, query_matches in all_matches.items():
                query_filename = os.path.basename(query_path)
                f.write(f"Query Image: {query_filename}\n")
                
                if not query_matches:
                    f.write("  No faces found or no matches found\n\n")
                    continue
                
                for face_match in query_matches:
                    face_idx = face_match['query_face_idx']
                    matches = face_match['matches']
                    
                    f.write(f"  Face #{face_idx+1}:\n")
                    
                    if not matches:
                        f.write("    No matches found\n")
                    else:
                        f.write(f"    Found {len(matches)} matches:\n")
                        
                        for i, match in enumerate(matches):
                            dataset_filename = os.path.basename(match['dataset_path'])
                            dataset_face_idx = match['dataset_face_idx']
                            similarity = match['similarity']
                            confidence_pct = int(similarity * 100)
                            
                            # Mark high confidence matches
                            high_conf_indicator = " [HIGH CONFIDENCE]" if similarity >= self.high_confidence_threshold else ""
                            
                            f.write(f"    {i+1}. {dataset_filename} (Face #{dataset_face_idx+1}): ")
                            f.write(f"Similarity = {confidence_pct}%{high_conf_indicator}\n")
                    
                    f.write("\n")
                
                f.write("--------------------------------------------------\n\n")
        
        print(f"Summary report saved to {report_path}")
    
    def _generate_high_confidence_report(self, output_folder):
        """Generate a report specifically for high confidence matches."""
        report_path = os.path.join(output_folder, "high_confidence_matches.txt")
        
        # Sort high confidence matches by similarity
        sorted_matches = sorted(self.high_confidence_matches, key=lambda x: x['similarity'], reverse=True)
        
        with open(report_path, 'w') as f:
            f.write("High Confidence Matches Report (75%+ Similarity)\n")
            f.write("=============================================\n\n")
            
            for i, match in enumerate(sorted_matches):
                query_filename = match['query_filename']
                dataset_filename = match['dataset_filename']
                similarity = match['similarity']
                confidence_pct = int(similarity * 100)
                
                f.write(f"Match #{i+1}: {confidence_pct}% Confidence\n")
                f.write(f"  Query: {query_filename} (Face #{match['query_face_idx']+1})\n")
                f.write(f"  Found in: {dataset_filename} (Face #{match['dataset_face_idx']+1})\n\n")
        
        print(f"\nHigh confidence report saved to {report_path}")
        return sorted_matches


def main():
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description='Match faces between query images and a dataset of images.')
    parser.add_argument('query_folder', nargs='?', default='queries',
                        help='Path to folder containing query images (default: "queries")')
    parser.add_argument('dataset_folder', nargs='?', default='dataset',
                        help='Path to folder containing dataset images to search through (default: "dataset")')
    parser.add_argument('--tolerance', type=float, default=0.6, 
                        help='Threshold for face matching (0.6 recommended, lower is stricter)')
    parser.add_argument('--high-confidence', type=float, default=0.75,
                        help='Threshold for high confidence matches (default: 0.75 = 75%%)')
    parser.add_argument('--output', default='results',
                        help='Path to folder where results will be saved')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png'],
                        help='Image extensions to process (default: .jpg .jpeg .png)')
    
    args = parser.parse_args()
    
    # Create the query and dataset folders if they don't exist
    os.makedirs(args.query_folder, exist_ok=True)
    os.makedirs(args.dataset_folder, exist_ok=True)
    
    # Check if folders are empty and provide guidance
    query_empty = not any(os.path.isfile(os.path.join(args.query_folder, f)) for f in os.listdir(args.query_folder))
    dataset_empty = not any(os.path.isfile(os.path.join(args.dataset_folder, f)) for f in os.listdir(args.dataset_folder))
    
    if query_empty or dataset_empty:
        print("\nFolder setup complete, but one or more folders are empty:")
        if query_empty:
            print(f"- Query folder '{args.query_folder}/' is empty. Please add images of faces you want to find.")
        if dataset_empty:
            print(f"- Dataset folder '{args.dataset_folder}/' is empty. Please add images to search through.")
        print("\nOnce you've added images to both folders, run this script again.")
        return
    
    # Initialize the face matcher with the specified tolerance
    matcher = FaceMatcher(tolerance=args.tolerance, high_confidence_threshold=args.high_confidence)
    
    try:
        # Match faces across folders
        all_matches = matcher.match_faces_across_folder(
            args.query_folder,
            args.dataset_folder,
            output_folder=args.output,
            image_extensions=args.extensions
        )
        
        # Print summary
        total_queries = len(all_matches)
        queries_with_matches = sum(1 for matches in all_matches.values() if any(face['matches'] for face in matches))
        
        print("\nMatching Complete!")
        print(f"Processed {total_queries} query images")
        print(f"Found matches for {queries_with_matches} query images")
        print(f"Results saved to: {os.path.abspath(args.output)}")
        
        # Display high confidence message
        if matcher.high_confidence_matches:
            high_conf_count = len(matcher.high_confidence_matches)
            print(f"\n╔{'═' * 60}╗")
            print(f"║ {' ' * 14}HIGH CONFIDENCE MATCHES FOUND!{' ' * 14} ║")
            print(f"║ {' ' * 4}Found {high_conf_count} matches with confidence above {int(args.high_confidence*100)}%{' ' * 4} ║")
            print(f"╚{'═' * 60}╝")
            print(f"\nCheck the high_confidence_matches.txt file for details.")
        
    except ValueError as e:
        # Handle the case where there are no images in the folders
        print(f"Error: {e}")
        print("\nPlease make sure you have:")
        print(f"1. At least one image in the query folder: {os.path.abspath(args.query_folder)}")
        print(f"2. At least one image in the dataset folder: {os.path.abspath(args.dataset_folder)}")
        print("3. Images with the supported extensions (.jpg, .jpeg, .png by default)")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
