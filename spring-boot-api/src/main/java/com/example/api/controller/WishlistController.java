package com.example.api.controller;

import com.example.api.model.Wishlist;
import com.example.api.service.WishlistService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/wishlist")
public class WishlistController {
    
    @Autowired
    private WishlistService wishlistService;
    
    @GetMapping
    public ResponseEntity<List<Wishlist>> getAllWishlistItems() {
        return ResponseEntity.ok(wishlistService.getAllWishlistItems());
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<Wishlist> getWishlistItemById(@PathVariable Long id) {
        return wishlistService.getWishlistItemById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/user/{userId}")
    public ResponseEntity<List<Wishlist>> getWishlistByUserId(@PathVariable Long userId) {
        return ResponseEntity.ok(wishlistService.getWishlistByUserId(userId));
    }
    
    @PostMapping
    public ResponseEntity<?> addToWishlist(@Valid @RequestBody Wishlist wishlist) {
        try {
            Wishlist addedWishlist = wishlistService.addToWishlist(wishlist);
            return ResponseEntity.status(HttpStatus.CREATED).body(addedWishlist);
        } catch (RuntimeException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<Wishlist> updateWishlistItem(@PathVariable Long id, @Valid @RequestBody Wishlist wishlist) {
        try {
            Wishlist updatedWishlist = wishlistService.updateWishlistItem(id, wishlist);
            return ResponseEntity.ok(updatedWishlist);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteWishlistItem(@PathVariable Long id) {
        wishlistService.deleteWishlistItem(id);
        return ResponseEntity.noContent().build();
    }
}
