package com.example.api.service;

import com.example.api.model.Review;
import com.example.api.repository.ReviewRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class ReviewService {
    
    @Autowired
    private ReviewRepository reviewRepository;
    
    public List<Review> getAllReviews() {
        return reviewRepository.findAll();
    }
    
    public Optional<Review> getReviewById(Long id) {
        return reviewRepository.findById(id);
    }
    
    public List<Review> getReviewsByProductId(Long productId) {
        return reviewRepository.findByProductId(productId);
    }
    
    public List<Review> getReviewsByUserId(Long userId) {
        return reviewRepository.findByUserId(userId);
    }
    
    public Review createReview(Review review) {
        return reviewRepository.save(review);
    }
    
    public Review updateReview(Long id, Review reviewDetails) {
        Review review = reviewRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Review not found with id: " + id));
        
        review.setRating(reviewDetails.getRating());
        review.setContent(reviewDetails.getContent());
        review.setTitle(reviewDetails.getTitle());
        
        return reviewRepository.save(review);
    }
    
    public void deleteReview(Long id) {
        reviewRepository.deleteById(id);
    }
}
