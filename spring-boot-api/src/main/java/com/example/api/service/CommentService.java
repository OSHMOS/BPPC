package com.example.api.service;

import com.example.api.model.Comment;
import com.example.api.repository.CommentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class CommentService {
    
    @Autowired
    private CommentRepository commentRepository;
    
    public List<Comment> getAllComments() {
        return commentRepository.findAll();
    }
    
    public Optional<Comment> getCommentById(Long id) {
        return commentRepository.findById(id);
    }
    
    public List<Comment> getCommentsByPostId(Long postId) {
        return commentRepository.findByPostId(postId);
    }
    
    public List<Comment> getCommentsByUserId(Long userId) {
        return commentRepository.findByUserId(userId);
    }
    
    public Comment createComment(Comment comment) {
        return commentRepository.save(comment);
    }
    
    public Comment updateComment(Long id, Comment commentDetails) {
        Comment comment = commentRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Comment not found with id: " + id));
        
        comment.setContent(commentDetails.getContent());
        return commentRepository.save(comment);
    }
    
    public void deleteComment(Long id) {
        commentRepository.deleteById(id);
    }
}
