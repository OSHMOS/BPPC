package com.example.api.service;

import com.example.api.model.Cart;
import com.example.api.repository.CartRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class CartService {
    
    @Autowired
    private CartRepository cartRepository;
    
    public List<Cart> getAllCartItems() {
        return cartRepository.findAll();
    }
    
    public Optional<Cart> getCartItemById(Long id) {
        return cartRepository.findById(id);
    }
    
    public List<Cart> getCartItemsByUserId(Long userId) {
        return cartRepository.findByUserId(userId);
    }
    
    public Cart addToCart(Cart cart) {
        Optional<Cart> existingCart = cartRepository.findByUserIdAndProductId(
                cart.getUserId(), cart.getProductId());
        
        if (existingCart.isPresent()) {
            Cart existing = existingCart.get();
            existing.setQuantity(existing.getQuantity() + cart.getQuantity());
            return cartRepository.save(existing);
        }
        
        return cartRepository.save(cart);
    }
    
    public Cart updateCartItem(Long id, Cart cartDetails) {
        Cart cart = cartRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Cart item not found with id: " + id));
        
        cart.setQuantity(cartDetails.getQuantity());
        return cartRepository.save(cart);
    }
    
    public void deleteCartItem(Long id) {
        cartRepository.deleteById(id);
    }
    
    public void clearCart(Long userId) {
        cartRepository.deleteByUserId(userId);
    }
}
