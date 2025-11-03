package com.example.api.controller;

import com.example.api.model.Cart;
import com.example.api.service.CartService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/cart")
public class CartController {
    
    @Autowired
    private CartService cartService;
    
    @GetMapping
    public ResponseEntity<List<Cart>> getAllCartItems() {
        return ResponseEntity.ok(cartService.getAllCartItems());
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<Cart> getCartItemById(@PathVariable Long id) {
        return cartService.getCartItemById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/user/{userId}")
    public ResponseEntity<List<Cart>> getCartItemsByUserId(@PathVariable Long userId) {
        return ResponseEntity.ok(cartService.getCartItemsByUserId(userId));
    }
    
    @PostMapping
    public ResponseEntity<Cart> addToCart(@Valid @RequestBody Cart cart) {
        Cart addedCart = cartService.addToCart(cart);
        return ResponseEntity.status(HttpStatus.CREATED).body(addedCart);
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<Cart> updateCartItem(@PathVariable Long id, @Valid @RequestBody Cart cart) {
        try {
            Cart updatedCart = cartService.updateCartItem(id, cart);
            return ResponseEntity.ok(updatedCart);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteCartItem(@PathVariable Long id) {
        cartService.deleteCartItem(id);
        return ResponseEntity.noContent().build();
    }
    
    @DeleteMapping("/user/{userId}/clear")
    public ResponseEntity<Void> clearCart(@PathVariable Long userId) {
        cartService.clearCart(userId);
        return ResponseEntity.noContent().build();
    }
}
