package com.example.api.service;

import com.example.api.model.Wishlist;
import com.example.api.repository.WishlistRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class WishlistService {
    
    @Autowired
    private WishlistRepository wishlistRepository;
    
    public List<Wishlist> getAllWishlistItems() {
        return wishlistRepository.findAll();
    }
    
    public Optional<Wishlist> getWishlistItemById(Long id) {
        return wishlistRepository.findById(id);
    }
    
    public List<Wishlist> getWishlistByUserId(Long userId) {
        return wishlistRepository.findByUserId(userId);
    }
    
    public Wishlist addToWishlist(Wishlist wishlist) {
        if (wishlistRepository.existsByUserIdAndProductId(
                wishlist.getUserId(), wishlist.getProductId())) {
            throw new RuntimeException("Product already in wishlist");
        }
        return wishlistRepository.save(wishlist);
    }
    
    public Wishlist updateWishlistItem(Long id, Wishlist wishlistDetails) {
        Wishlist wishlist = wishlistRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Wishlist item not found with id: " + id));
        
        wishlist.setNotes(wishlistDetails.getNotes());
        wishlist.setPriority(wishlistDetails.getPriority());
        wishlist.setNotifyOnSale(wishlistDetails.getNotifyOnSale());
        
        return wishlistRepository.save(wishlist);
    }
    
    public void deleteWishlistItem(Long id) {
        wishlistRepository.deleteById(id);
    }
}
